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
args.model_path = 'F:/Projects/OutfitTransformer/checkpoints/cp/240601/cp_best_model_ep7_acc0.838.pth'
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
args.num_workers = 0
args.test_batch_size = 128
args.with_cuda = True


def cp_evaluation(model, dataloader, device):
    type_str = f'cp test'
    epoch_iterator = tqdm(dataloader)
    
    total_y_true = []
    total_y_pred = []
    
    for iter, batch in enumerate(epoch_iterator, start=1):
        targets = batch['targets'].to(device)
        inputs = {key: value.to(device) for key, value in batch['inputs'].items()}

        input_embeddings = model.batch_encode(inputs)            
        logits = model.calculate_compatibility(input_embeddings)

        total_y_true.append(targets.clone().flatten().detach().cpu().bool())
        total_y_pred.append((logits > 0.5).flatten().detach().cpu().bool())
        is_correct = (total_y_true[-1] == total_y_pred[-1])
        running_acc = torch.sum(is_correct).item() / torch.numel(is_correct)

        epoch_iterator.set_description(
            f'Acc: {running_acc:.3f}')
        
    total_y_true = torch.concat(total_y_true)
    total_y_pred = torch.concat(total_y_pred)
    is_correct = (total_y_true == total_y_pred)
    acc = torch.sum(is_correct).item() / torch.numel(is_correct)
    auc = roc_auc_score(total_y_true, total_y_pred)
    print( f'[{type_str} END] | Acc: {acc:.3f} | AUC: {auc:.3f}' + '\n')


if __name__ == '__main__':
    TASK = 'cp'

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
        cp_evaluation(model, test_dataloader, device)
