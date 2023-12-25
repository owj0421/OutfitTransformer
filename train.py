import os
import wandb
import argparse
from itertools import chain

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.model import *
from model.encoder import ItemEncoder
from utils.trainer import Trainer, TrainingArgs
from utils.dataset import PolyvoreDataset, DatasetArgs
from utils.metric import MetricCalculator

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Parser
parser = argparse.ArgumentParser(description='Outfit-Transformer Trainer')
parser.add_argument('--task', help='compatibility, fitb', type=str, default='compatibility')
parser.add_argument('--train_batch', help='Size of Batch for Training', type=int, default=48)
parser.add_argument('--valid_batch', help='Size of Batch for Validation, Test', type=int, default=64)
parser.add_argument('--n_epochs', help='Number of epochs', type=int, default=5)
parser.add_argument('--scheduler_step_size', help='Step LR', type=int, default=1000)
parser.add_argument('--learning_rate', help='Learning rate', type=float, default=1e-5)
parser.add_argument('--work_dir', help='Full working directory', type=str, required=True)
parser.add_argument('--data_dir', help='Full dataset directory', type=str, required=True)
parser.add_argument('--wandb_api_key', required=True)
parser.add_argument('--checkpoint', default=None)
args = parser.parse_args()

# Wandb
os.environ["WANDB_API_KEY"] = args.wandb_api_key
os.environ["WANDB_PROJECT"] = f"outfit-transformer-{args.task}"
os.environ["WANDB_LOG_MODEL"] = "all"
wandb.login()
run = wandb.init()

# Setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

training_args = TrainingArgs(
    task=args.task,
    train_batch=args.train_batch,
    valid_batch=args.valid_batch,
    n_epochs=args.n_epochs,
    learning_rate=args.learning_rate,
    work_dir = args.work_dir
    )
    
dataset_args = DatasetArgs(
    work_dir=args.work_dir, 
    data_dir=args.data_dir
    )

train_transform = A.Compose([
    A.Resize(230, 230),
    A.HorizontalFlip(),
    A.RandomResizedCrop(scale=(0.85, 1.05), height=dataset_args.img_size, width=dataset_args.img_size, always_apply=True, p=1),
    A.Rotate(limit=15, p=0.5),
    A.Normalize(),
    ToTensorV2()
    ])

train_dataset = PolyvoreDataset(dataset_args, task=training_args.task, dataset_type='train', train_transform=train_transform)
valid_dataset = PolyvoreDataset(dataset_args, task=training_args.task, dataset_type='valid', train_transform=train_transform)
train_dataloader = DataLoader(train_dataset, training_args.train_batch, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, training_args.valid_batch, shuffle=False)
print('[COMPLETE] Build Dataset')
encoder = ItemEncoder(embedding_dim=128).to(device)
print('[COMPLETE] Build Encoder')
model = OutfitTransformer(embedding_dim=128).to(device)
print('[COMPLETE] Build Model')

optimizer = AdamW(chain(model.parameters(), encoder.parameters()), lr=training_args.learning_rate,)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.5)
metric = MetricCalculator()
trainer = Trainer(model, encoder, train_dataloader, valid_dataloader,
                  optimizer=optimizer, scheduler=scheduler, metric=metric, device=device, args=training_args)

if args.checkpoint != None:
    checkpoint = args.checkpoint
    trainer.load(checkpoint, load_optim=False)

# Train
trainer.train()