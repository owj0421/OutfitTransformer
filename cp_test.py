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

args.data_dir = '/home/datasets/polyvore_outfits'
args.checkpoint_dir = './checkpoints'
args.model_path = './checkpoints/outfit_transformer/cp/240608/AUC0.908.pth'

# Training Setting
args.num_workers = 0
args.test_batch_size = 8
args.with_cuda = True


def cp_evaluation(model,dataloader, device):
    type_str = f'cp test'
    epoch_iterator = tqdm(dataloader)
    
    total_y_true = []
    total_y_score = []
    
    for iter, batch in enumerate(epoch_iterator, start=1):
        with torch.cuda.amp.autocast():
            targets = batch['targets'].to(device)
            inputs = {key: value.to(device) for key, value in batch['inputs'].items()}

            input_embeddings = model.batch_encode(inputs)            
            probs = model.get_score(input_embeddings)

        total_y_true.append(targets.clone().flatten().detach().cpu().bool())
        total_y_score.append(probs.flatten().detach().cpu())
        is_correct = (total_y_true[-1] == (total_y_score[-1] > 0.5))
        running_acc = torch.sum(is_correct).item() / torch.numel(is_correct)

        epoch_iterator.set_description(
            f'Acc: {running_acc:.3f}')

    total_y_true = torch.concat(total_y_true)
    total_y_score = torch.concat(total_y_score)
    is_correct = (total_y_true == (total_y_score > 0.5))
    acc = torch.sum(is_correct).item() / torch.numel(is_correct)
    try:
        auc = roc_auc_score(total_y_true, total_y_score)
    except:
        auc = 0
    print( f'[{type_str} END] Acc: {acc:.3f} | AUC: {auc:.3f}' + '\n')


if __name__ == '__main__':
    cuda_condition = torch.cuda.is_available() and args.with_cuda
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    
    model, input_processor = load_model(args)
    model.to(device)

    test_dataset_args = DatasetArguments(
        polyvore_split=args.polyvore_split, task_type='cp', dataset_type='test')
    test_dataset = PolyvoreDataset(args.data_dir, test_dataset_args, input_processor)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    model.eval()
    with torch.no_grad():
        cp_evaluation(model, dataloader=test_dataloader, device=device)