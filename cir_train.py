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
args.model_path = './checkpoints/outfit_transformer/cir/240610/ACC0.647.pth'
    
# Training Setting
args.n_epochs = 3
args.num_workers = 0
args.train_batch_size = 64
args.val_batch_size = 64
args.lr = 4e-5
args.wandb_key = None
args.use_wandb = True if args.wandb_key else False
args.with_cuda = True


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
        
        # query_inputs = {
        #     'mask': positive['mask'],
        #     'input_ids': positive['input_ids'],
        #     'attention_mask': positive['attention_mask']
        #     }

        anchor_item_embeds = model.batch_encode(anchor)
        anchor_embeds = model.get_embedding(anchor_item_embeds) # , query_inputs) # B, EMBEDDING_DIM
        
        positive_item_embeds = model.batch_encode(positive)
        positive_embeds = model.get_embedding(positive_item_embeds) # B, EMBEDDING_DIM

        running_loss = criterion(
            query=anchor_embeds,
            positive_key=positive_embeds,
            )
        
        loss += running_loss.item()
        if is_train == True:
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        with torch.no_grad():
            logits = anchor_embeds @ positive_embeds.transpose(-2, -1)
            labels = torch.arange(len(anchor_embeds), device=anchor_embeds.device)
            
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


def fitb_iteration(epoch, model, optimizer, scheduler, dataloader, device, is_train, use_wandb):
    criterion = InfoNCE(negative_mode='paired')
    
    type_str = f'fitb train' if is_train else f'fitb valid'
    epoch_iterator = tqdm(dataloader)
    
    loss = 0.
    total_y_true = []
    total_y_pred = []
    
    for iter, batch in enumerate(epoch_iterator, start=1):
        questions = {key: value.to(device) for key, value in batch['questions'].items()}
        candidates = {key: value.to(device) for key, value in batch['candidates'].items()}

        question_item_embeds = model.batch_encode(questions)           
        question_embeds = model.get_embedding(question_item_embeds) # B, EMBEDDING_DIM
        
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
        if is_train == True:
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

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
    print( f'[{type_str} END] Epoch: {epoch + 1:03} | loss: {loss:.5f} | Acc: {acc:.3f}')

    return loss, acc


if __name__ == '__main__':
    TASK = 'cir'
    EMBEDDER_TYPE = 'outfit_transformer' if not args.use_clip_embedding else 'clip'

    # Wandb
    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = args.wandb_key
        os.environ["WANDB_PROJECT"] = f"OutfitTransformer-{TASK}"
        os.environ["WANDB_LOG_MODEL"] = "all"
        wandb.login()
        run = wandb.init()
    
    date_info = datetime.today().strftime("%y%m%d")
    save_dir = os.path.join(args.checkpoint_dir, EMBEDDER_TYPE, TASK, date_info)

    cuda_condition = torch.cuda.is_available() and args.with_cuda
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    
    model, input_processor = load_model(args)
    model.to(device)

    # train_dataset_args = DatasetArguments(
    #     polyvore_split=args.polyvore_split, task_type='cir', dataset_type='train')
    train_dataset_args = DatasetArguments(
        polyvore_split=args.polyvore_split, task_type='cir', dataset_type='train')
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
        # train_loss, train_acc = cir_iteration(
        #     epoch, model, optimizer, scheduler,
        #     dataloader=train_dataloader, device=device, is_train=True, use_wandb=args.use_wandb
        #     )
        train_loss, train_acc = cir_iteration(
            epoch, model, optimizer, scheduler,
            dataloader=train_dataloader, device=device, is_train=True, use_wandb=args.use_wandb
            )
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = fitb_iteration(
                epoch, model, optimizer, scheduler,
                dataloader=val_dataloader, device=device, is_train=False, use_wandb=args.use_wandb
                )
        if val_acc >= best_acc:
            best_acc = val_acc
            model_name = f'ACC{val_acc:.3f}'
            save_model(model, save_dir, model_name, device)