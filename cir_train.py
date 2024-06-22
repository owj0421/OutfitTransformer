import torch
from torch.utils.data import DataLoader

from src.datasets.polyvore import DatasetArguments, PolyvoreDataset

from src.models.embedder import CLIPEmbeddingModel
from src.models.recommender import RecommendationModel
from src.models.load import load_model
from src.loss.info_nce import InfoNCE
from src.utils.utils import save_model

import os
import wandb
import numpy as np

from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score

from model_args import Args
args = Args()

args.data_dir = '/home/datasets/polyvore_outfits'
args.checkpoint_dir = './checkpoints'
args.model_path = '/home/OutfitTransformer/checkpoints/outfit_transformer/cir/240610/fitb_acc0.64.pth'
    
# Training Setting
args.n_epochs = 15
args.num_workers = 0
args.train_batch_size = 128
args.val_batch_size = 128
args.lr = 4e-5
args.wandb_key = 'fa37a3c4d1befcb0a7b9b4d33799c7bdbff1f81f'
args.use_wandb = True if args.wandb_key else False
args.with_cuda = True

def transpose(x):
    return x.transpose(-2, -1)


def cir_iteration(epoch, model, optimizer, scheduler, dataloader, device, is_train, use_wandb):
    criterion = InfoNCE(negative_mode='unpaired')
    
    type_str = f'cir train' if is_train else f'cir valid'
    epoch_iterator = tqdm(dataloader)
    
    loss = 0.
    total_y_true = []
    total_y_pred = []
    
    for iter, batch in enumerate(epoch_iterator, start=1):
        anchor = {key: value.to(device) for key, value in batch['anchor'].items()}
        positive = {key: value.to(device) for key, value in batch['positive'].items()}
        negative = {key: value.to(device) for key, value in batch['negative'].items()}
        
        query = {
            'input_ids': positive['category_input_ids'].squeeze(1),
            'attention_mask': positive['category_attention_mask'].squeeze(1)
            }

        anchor_item_embeds = model.batch_encode(anchor)
        anchor_embeds = model.get_embedding(anchor_item_embeds, query) # , query_inputs) # B, EMBEDDING_DIM
        
        positive_item_embeds = model.batch_encode(positive)
        positive_embeds = model.get_embedding(positive_item_embeds) # B, EMBEDDING_DIM
        
        negative_item_embeds = model.batch_encode(negative)
        negative_embeds = model.get_embedding(negative_item_embeds)  # B, N_NEGATIVES, EMBEDDING_DIM
        
        # Compute Loss
        logits_pospos = anchor_embeds @ transpose(positive_embeds)
        logits_posneg = (anchor_embeds.unsqueeze(1) @ transpose(negative_embeds)).squeeze(1)
        logits = torch.cat((logits_pospos, logits_posneg), dim=1)
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        running_loss = torch.nn.functional.cross_entropy(logits / 0.1, labels, reduction='mean')
        
        loss += running_loss.item()
        if is_train == True:
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        with torch.no_grad():
            total_y_true.append(labels.cpu())
            total_y_pred.append(torch.argmax(logits, dim=-1).cpu())

        is_correct = (total_y_true[-1] == total_y_pred[-1])
        running_acc = torch.sum(is_correct).item() / torch.numel(is_correct)

        epoch_iterator.set_description(
            f'Loss: {running_loss:.5f} | Acc: {running_acc:.3f}')
        if use_wandb:
            log = {
                f'{type_str}_loss': running_loss, 
                f'{type_str}_acc': running_acc, 
                f'{type_str}_step': epoch * len(epoch_iterator) + iter
                }
            if (is_train == True) and (scheduler is not None):
                log["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log)

    loss = loss / iter
    total_y_true = torch.cat(total_y_true)
    total_y_pred = torch.cat(total_y_pred)
    is_correct = (total_y_true == total_y_pred)
    acc = torch.sum(is_correct).item() / torch.numel(is_correct)
    print( f'[{type_str} END] Epoch: {epoch + 1:03} | loss: {loss:.5f} | Acc: {acc:.3f}')

    return loss, acc


def fitb_evaluation(epoch, model, optimizer, scheduler, dataloader, device, is_train, use_wandb):
    criterion = InfoNCE(negative_mode='paired')
    
    type_str = f'fitb test'
    epoch_iterator = tqdm(dataloader)
    
    loss = 0.
    total_y_true = []
    total_y_pred = []
    
    for iter, batch in enumerate(epoch_iterator, start=1):
        questions = {key: value.to(device) for key, value in batch['questions'].items()}
        candidates = {key: value.to(device) for key, value in batch['candidates'].items()}
        query = {
            'input_ids': candidates['category_input_ids'][:, 0],
            'attention_mask': candidates['category_attention_mask'][:, 0]
        }
        question_item_embeds = model.batch_encode(questions)           
        question_embeds = model.get_embedding(question_item_embeds, query) # B, EMBEDDING_DIM 
        
        candidate_item_embeds = model.batch_encode(candidates) # B, N_CANDIDATES(1 positive, 3 negative), EMBEDDING_DIM
        B, N_CANDIDATES = candidates['mask'].shape
        candidate_item_embeds['mask'] = candidate_item_embeds['mask'].view(B * N_CANDIDATES, -1)
        candidate_item_embeds['embeds'] = candidate_item_embeds['embeds'].view(B * N_CANDIDATES, 1, -1)
        
        candidate_embeds = model.get_embedding(candidate_item_embeds).view(B, N_CANDIDATES, -1) # B, N_CANDIDATES, EMBEDDING_DIM
        pos_candidate_embeds = candidate_embeds[:, 0, :]
        neg_candidate_embeds = candidate_embeds[:, 1:, :]

        running_loss = criterion(
            query=question_embeds, 
            positive_key=pos_candidate_embeds,
            negative_keys=neg_candidate_embeds
            )
        
        loss += running_loss.item()

        with torch.no_grad():
            scores = torch.sum(question_embeds.unsqueeze(1).detach() * candidate_embeds.detach(), dim=-1)
            total_y_true.append(torch.zeros(B, dtype=torch.long).cpu())
            total_y_pred.append(torch.argmax(scores, dim=-1).cpu())

        is_correct = (total_y_true[-1] == total_y_pred[-1])
        running_acc = torch.sum(is_correct).item() / torch.numel(is_correct)

        epoch_iterator.set_description(
            f'Loss: {running_loss:.5f} | Acc: {running_acc:.3f}')
        if use_wandb:
            log = {
                f'{type_str}_loss': running_loss, 
                f'{type_str}_acc': running_acc, 
                f'{type_str}_step': epoch * len(epoch_iterator) + iter
                }
            if (is_train == True) and (scheduler is not None):
                log["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log)

    loss = loss / iter
    total_y_true = torch.cat(total_y_true)
    total_y_pred = torch.cat(total_y_pred)
    is_correct = (total_y_true == total_y_pred)
    acc = torch.sum(is_correct).item() / torch.numel(is_correct)
    print( f'[{type_str} END] loss: {loss:.5f} | Acc: {acc:.3f}')

    return loss, acc


if __name__ == '__main__':
    train_task = 'cir'
    EMBEDDER_TYPE = 'outfit_transformer' if not args.use_clip_embedding else 'clip'

    # Wandb
    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = args.wandb_key
        os.environ["WANDB_PROJECT"] = f"OutfitTransformer-{train_task}"
        os.environ["WANDB_LOG_MODEL"] = "all"
        wandb.login()
        run = wandb.init()
    
    date_info = datetime.today().strftime("%y%m%d")
    save_dir = os.path.join(args.checkpoint_dir, EMBEDDER_TYPE, train_task, date_info)

    cuda_condition = torch.cuda.is_available() and args.with_cuda
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    
    model, input_processor = load_model(args)
    model.to(device)
    
    train_dataset_args = DatasetArguments(
        polyvore_split=args.polyvore_split, task_type=train_task, dataset_type='train')
    train_dataset = PolyvoreDataset(args.data_dir, train_dataset_args, input_processor)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    
    val_dataset_args = DatasetArguments(
        polyvore_split=args.polyvore_split, task_type='fitb', dataset_type='valid')
    val_dataset = PolyvoreDataset(args.data_dir, val_dataset_args, input_processor)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(optimizer, args.lr, epochs=args.n_epochs, steps_per_epoch=len(train_dataloader))

    best_acc = 0
    for epoch in range(args.n_epochs):
        model.train()
        train_loss, train_acc = cir_iteration(
            epoch, model, optimizer, scheduler,
            dataloader=train_dataloader, device=device, is_train=True, use_wandb=args.use_wandb
            )
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = fitb_evaluation(
                epoch, model, optimizer, scheduler,
                dataloader=val_dataloader, device=device, is_train=False, use_wandb=args.use_wandb
                )
        if val_acc >= best_acc:
            best_acc = val_acc
            model_name = f'cir_acc{val_acc:.3f}'
            save_model(model, save_dir, model_name, device)