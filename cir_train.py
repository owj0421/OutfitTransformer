import torch
from torch.utils.data import DataLoader

from src.datasets.polyvore import DatasetArguments, PolyvoreDataset

from src.models.embedder import CLIPEmbeddingModel
from src.models.recommender import RecommendationModel
from src.models.load import load_model
from src.loss.focal_loss import focal_loss
from src.utils.utils import save_model

import os
import wandb
import numpy as np

from bitsandbytes.optim import AdamW8bit
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score

@ dataclass
class Args:
    pass


args = Args()
# Dir Settings
args.data_dir = 'F:/Projects/datasets/polyvore_outfits'
args.checkpoint_dir = 'F:/Projects/OutfitTransformer/checkpoints'
args.model_path = None
args.load_model = True if args.model_path is not None else False
# Dataset & Input Processor Settings
args.polyvore_split = 'nondisjoint'
args.categories = ['<bottoms>', '<outerwear>', '<tops>', '<scarves>', '<hats>', '<all-body>', '<accessories>', '<sunglasses>', '<shoes>', '<jewellery>', '<bags>']
args.outfit_max_length = 16
args.use_image = True
args.use_text = True
args.use_category = False
args.text_max_length = 16
# Embedder&Recommender Model Settings
args.use_clip_embedding = False
args.clip_huggingface = 'patrickjohncyh/fashion-clip'
args.huggingface = 'sentence-transformers/all-MiniLM-L6-v2'
args.hidden = 128
args.n_layers = 6
args.n_heads = 16

# Training Setting
args.n_epochs = 9
args.num_workers = 0
args.train_batch_size = 50
args.val_batch_size = 50
args.lr = 1e-4
args.wandb_key = None
args.use_wandb = True if args.wandb_key else False
args.with_cuda = True


def cp_iteration(epoch, model, optimizer, scheduler, criterion, dataloader, device, is_train, use_wandb):
    scaler = torch.cuda.amp.GradScaler()

    type_str = f'cp train' if is_train else f'cp valid'
    epoch_iterator = tqdm(dataloader)
    
    loss = 0.
    total_y_true = []
    total_y_pred = []
    
    for iter, batch in enumerate(epoch_iterator, start=1):
        with torch.cuda.amp.autocast():
            targets = batch['targets'].to(device)
            inputs = {key: value.to(device) for key, value in batch['inputs'].items()}

            input_embeddings = model.batch_encode(inputs)            
            logits = model.calculate_compatibility(input_embeddings)

        running_loss = criterion(logits, targets)
        loss += running_loss.item()
        if is_train == True:
            optimizer.zero_grad()
            scaler.scale(running_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.step()
            if scheduler:
                scheduler.step()

        total_y_true.append(targets.clone().flatten().detach().cpu().bool())
        total_y_pred.append((logits > 0.5).flatten().detach().cpu().bool())
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
    total_y_true = torch.concat(total_y_true)
    total_y_pred = torch.concat(total_y_pred)
    is_correct = (total_y_true == total_y_pred)
    acc = torch.sum(is_correct).item() / torch.numel(is_correct)
    auc = roc_auc_score(total_y_true, total_y_pred)
    print( f'[{type_str} END] Epoch: {epoch + 1:03} | loss: {loss:.5f} | Acc: {acc:.3f} | AUC: {auc:.3f}' + '\n')

    return loss, acc


if __name__ == '__main__':
    TASK = 'cp'

    # Wandb
    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = args.wandb_key
        os.environ["WANDB_PROJECT"] = f"OutfitTransformer-{TASK}"
        os.environ["WANDB_LOG_MODEL"] = "all"
        wandb.login()
        run = wandb.init()
    
    date_info = datetime.today().strftime("%y%m%d")
    save_dir = os.path.join(args.checkpoint_dir, TASK, date_info)

    cuda_condition = torch.cuda.is_available() and args.with_cuda
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    
    model, input_processor = load_model(args)
    model.to(device)
    
    train_dataset_args = DatasetArguments(
        polyvore_split=args.polyvore_split, task_type=TASK, dataset_type='train')
    train_dataset = PolyvoreDataset(args.data_dir, train_dataset_args, input_processor)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

    val_dataset_args = DatasetArguments(
        polyvore_split=args.polyvore_split, task_type=TASK, dataset_type='valid')
    val_dataset = PolyvoreDataset(args.data_dir, val_dataset_args, input_processor)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(optimizer, args.lr, epochs=args.n_epochs, steps_per_epoch=len(train_dataloader))
    criterion = focal_loss

    best_acc = 0
    for epoch in range(args.n_epochs):
        model.train()
        train_loss, train_acc = cp_iteration(
            epoch, model, optimizer, scheduler, criterion, 
            dataloader=train_dataloader, device=device, is_train=True, use_wandb=args.use_wandb
            )
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = cp_iteration(
                epoch, model, optimizer, scheduler, criterion, 
                dataloader=val_dataloader, device=device, is_train=False, use_wandb=args.use_wandb
                )
        if val_acc > best_acc:
            best_acc = val_acc
            model_name = f'{TASK}_best_model_ep{epoch}_acc{best_acc:.3f}'
            save_model(model, save_dir, model_name, device)