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
args.num_workers = 0
args.test_batch_size = 8
args.with_cuda = True


def fitb_evaluation(model, dataloader, device):
    criterion = InfoNCE(negative_mode='paired')
    
    type_str = f'fitb test'
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

        with torch.no_grad():
            scores = torch.sum(question_embeds.unsqueeze(1).detach() * candidate_embeds.detach(), dim=-1)
            total_y_true.append(torch.zeros(B, dtype=torch.long).cpu())
            total_y_pred.append(torch.argmax(scores, dim=-1).cpu())

        is_correct = (total_y_true[-1] == total_y_pred[-1])
        running_acc = torch.sum(is_correct).item() / torch.numel(is_correct)

        epoch_iterator.set_description(
            f'Loss: {running_loss:.5f} | Acc: {running_acc:.3f}')

    loss = loss / iter
    total_y_true = torch.cat(total_y_true)
    total_y_pred = torch.cat(total_y_pred)
    is_correct = (total_y_true == total_y_pred)
    acc = torch.sum(is_correct).item() / torch.numel(is_correct)
    print( f'[{type_str} END] loss: {loss:.5f} | Acc: {acc:.3f}')

    return loss, acc


if __name__ == '__main__':
    EMBEDDER_TYPE = 'outfit_transformer' if not args.use_clip_embedding else 'clip'

    cuda_condition = torch.cuda.is_available() and args.with_cuda
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    
    model, input_processor = load_model(args)
    model.to(device)
    
    test_dataset_args = DatasetArguments(
        polyvore_split=args.polyvore_split, task_type='fitb', dataset_type='test')
    test_dataset = PolyvoreDataset(args.data_dir, test_dataset_args, input_processor)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    model.eval()
    with torch.no_grad():
        fitb_evaluation(model, dataloader=test_dataloader, device=device)