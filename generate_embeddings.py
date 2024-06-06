import torch
from torch.utils.data import DataLoader

from src.datasets.polyvore_embed import DatasetArguments, PolyvoreDataset

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
import pickle

from model_args import Args
args = Args()

# Embedding Setting
args.num_workers = 4
args.batch_size = 512
args.with_cuda = True


def generate(model, dataloader, device):
    type_str = f'cp test'
    epoch_iterator = tqdm(dataloader)
    
    allids, allembeddings = [], []
    with torch.no_grad():
        for iter, batch in enumerate(epoch_iterator, start=1):
            item_ids, inputs = batch

            allids.extend(item_ids)
            with torch.cuda.amp.autocast():
                inputs = {key: value.to(device) for key, value in inputs.items()}
                _, batch_embeddings = model.encode(inputs).values()
            batch_embeddings = batch_embeddings.cpu().float()
            allembeddings.append(batch_embeddings)

    allembeddings = torch.cat(allembeddings, dim=0).numpy()

    return allids, allembeddings


if __name__ == '__main__':
    cuda_condition = torch.cuda.is_available() and args.with_cuda
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    
    model, input_processor = load_model(args)
    model.to(device)
    model = model.half()

    dataset = PolyvoreDataset(args.data_dir, args, input_processor)
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model.eval()
    with torch.no_grad():
        allids, allembeddings = generate(model, dataloader=dataloader, device=device)

    save_file = os.path.join('./', 'embeddings.pkl')
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    print(f"Total passages processed {len(allids)}. Written to {save_file}.")