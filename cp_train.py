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

from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score

from model_args import Args
args = Args()

# Training Setting
args.n_epochs = 100
args.num_workers = 4
args.train_batch_size = 512
args.val_batch_size = 1024
args.test_batch_size = 1024
args.lr = 4e-5
args.wandb_key = 'fa37a3c4d1befcb0a7b9b4d33799c7bdbff1f81f'
args.use_wandb = True if args.wandb_key else False
args.with_cuda = True


def cp_iteration(epoch, model, optimizer, scheduler, dataloader, device, is_train, use_wandb):
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
            probs = model.calculate_compatibility(input_embeddings)

        running_loss = focal_loss(probs, targets)
        loss += running_loss.item()
        if is_train == True:
            optimizer.zero_grad()
            scaler.scale(running_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()

        total_y_true.append(targets.clone().flatten().detach().cpu().bool())
        total_y_pred.append((probs > 0.5).flatten().detach().cpu().bool())
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
    try:
        auc = roc_auc_score(total_y_true, total_y_pred)
    except:
        auc = 0
    print( f'[{type_str} END] Epoch: {epoch + 1:03} | loss: {loss:.5f} | Acc: {acc:.3f} | AUC: {auc:.3f}' + '\n')

    return loss, acc, auc


if __name__ == '__main__':
    TASK = 'cp'
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

    best_auc = 0
    for epoch in range(args.n_epochs):
        model.train()
        train_loss, train_acc, train_auc = cp_iteration(
            epoch, model, optimizer, scheduler,
            dataloader=train_dataloader, device=device, is_train=True, use_wandb=args.use_wandb
            )
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_auc = cp_iteration(
                epoch, model, optimizer, scheduler,
                dataloader=val_dataloader, device=device, is_train=False, use_wandb=args.use_wandb
                )
        if val_auc >= best_auc:
            best_auc = val_auc
            model_name = f'AUC{best_auc:.3f}'
            save_model(model, save_dir, model_name, device)