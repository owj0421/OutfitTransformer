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
from utils.trainer import *
from utils.dataset import *
from utils.metric import MetricCalculator

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Parser
parser = argparse.ArgumentParser(description='Outfit-Transformer Trainer')
parser.add_argument('--train_task', help='cp, sampling', type=str, default='cp')
parser.add_argument('--valid_task', help='cp, fitb, sampling', type=str, default='fitb')
parser.add_argument('--train_batch', help='Size of Batch for Training', type=int, default=48)
parser.add_argument('--valid_batch', help='Size of Batch for Validation, Test', type=int, default=64)
parser.add_argument('--n_epochs', help='Number of epochs', type=int, default=5)
parser.add_argument('--scheduler_step_size', help='Step LR', type=int, default=1000)
parser.add_argument('--learning_rate', help='Learning rate', type=float, default=1e-5)
parser.add_argument('--work_dir', help='Full working directory', type=str, required=True)
parser.add_argument('--data_dir', help='Full dataset directory', type=str, required=True)
parser.add_argument('--wandb_api_key', default=None)
parser.add_argument('--checkpoint', default=None)
args = parser.parse_args()

# Wandb
if args.wandb_api_key:
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_PROJECT"] = f"outfit-transformer-{args.valid_task}"
    os.environ["WANDB_LOG_MODEL"] = "all"
    wandb.login()
    run = wandb.init()

# Setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

training_args = TrainingArguments(
    train_task=args.train_task,
    valid_task=args.valid_task,
    train_batch=args.train_batch,
    valid_batch=args.valid_batch,
    n_epochs=args.n_epochs,
    learning_rate=args.learning_rate,
    work_dir = args.work_dir,
    use_wandb = True if args.wandb_api_key else False
    )
    
train_dataset_args = DatasetArguments(
    polyvore_split = 'nondisjoint',
    task_type = args.train_task,
    dataset_type = 'train',
    img_size = 224,
    use_pretrined_tokenizer = True,
    max_token_len = 16,
    custom_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(),
        #A.RandomResizedCrop(scale=(0.90, 1.00), height=224, width=224, always_apply=True, p=1),
        #A.Rotate(limit=15, p=0.5),
        A.Normalize(),
        ToTensorV2()
        ])
    )

valid_dataset_args = DatasetArguments(
    polyvore_split = 'nondisjoint',
    task_type = args.valid_task,
    dataset_type = 'valid',
    img_size = 224,
    use_pretrined_tokenizer = True,
    max_token_len = 16,
    custom_transform = None
    )

train_dataset = PolyvoreDataset(args.work_dir, args.data_dir, train_dataset_args)
valid_dataset = PolyvoreDataset(args.work_dir, args.data_dir, valid_dataset_args)
train_dataloader = DataLoader(train_dataset, training_args.train_batch, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, training_args.valid_batch, shuffle=False)
print('[COMPLETE] Build Dataset, DataLoader')

encoder = ItemEncoder(embedding_dim=128).to(device)
print('[COMPLETE] Build Encoder')

model = OutfitTransformer(embedding_dim=128).to(device)
print('[COMPLETE] Build Model')

optimizer = AdamW(chain(model.parameters(), encoder.parameters()), lr=training_args.learning_rate,)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.5)
metric = MetricCalculator()
trainer = Trainer(training_args, model, encoder, train_dataloader, valid_dataloader,
                  optimizer=optimizer, metric=metric , scheduler=scheduler)

if args.checkpoint != None:
    checkpoint = args.checkpoint
    trainer.load(checkpoint, load_optim=False)
    print(f'[COMPLETE] Load Model from {checkpoint}')

# Train
trainer.fit()