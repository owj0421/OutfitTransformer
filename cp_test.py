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

from model_args import Args
args = Args()

# Training Setting
args.n_epochs = 10
args.num_workers = 0
args.train_batch_size = 64
args.val_batch_size = 96
args.test_batch_size = 96
args.lr = 1e-4
args.wandb_key = None
args.use_wandb = True if args.wandb_key else False
args.with_cuda = True


def cp_evaluation(model,dataloader, device, use_wandb):
    type_str = f'cp test'
    epoch_iterator = tqdm(dataloader)
    
    total_y_true = []
    total_y_pred = []
    
    for iter, batch in enumerate(epoch_iterator, start=1):
        with torch.cuda.amp.autocast():
            targets = batch['targets'].to(device)
            inputs = {key: value.to(device) for key, value in batch['inputs'].items()}

            input_embeddings = model.batch_encode(inputs)            
            probs = model.calculate_compatibility(input_embeddings)

        total_y_true.append(targets.clone().flatten().detach().cpu().bool())
        total_y_pred.append((probs > 0.5).flatten().detach().cpu().bool())
        is_correct = (total_y_true[-1] == total_y_pred[-1])
        running_acc = torch.sum(is_correct).item() / torch.numel(is_correct)

        epoch_iterator.set_description(
            f'Acc: {running_acc:.3f}')
        if use_wandb:
            log = {
                f'{type_str}_acc': running_acc, 
                }
            wandb.log(log)

    total_y_true = torch.concat(total_y_true)
    total_y_pred = torch.concat(total_y_pred)
    is_correct = (total_y_true == total_y_pred)
    acc = torch.sum(is_correct).item() / torch.numel(is_correct)
    auc = roc_auc_score(total_y_true, total_y_pred)
    print( f'[{type_str} END] Acc: {acc:.3f} | AUC: {auc:.3f}' + '\n')


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

    test_dataset_args = DatasetArguments(
        polyvore_split=args.polyvore_split, task_type=TASK, dataset_type='test')
    test_dataset = PolyvoreDataset(args.data_dir, test_dataset_args, input_processor)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)


    model.eval()
    with torch.no_grad():
        cp_evaluation(model, dataloader=test_dataloader, device=device, use_wandb=args.use_wandb)