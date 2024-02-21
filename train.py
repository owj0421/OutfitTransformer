# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import os
import wandb
import argparse
from itertools import chain

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from outfit_transformer.models.outfit_transformer import OutfitTransformer

from outfit_transformer.datasets.polyvore import *

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='OutfitTransformer')
    parser.add_argument('--embedding_dim', help='embedding dim', type=int, default=128)
    parser.add_argument('--task', help='', type=str, default='cp')

    parser.add_argument('--outfit_max_length', help='outfit_max_length', type=int, default=12)

    parser.add_argument('--train_batch', help='Size of Batch for Training', type=int, default=32)
    parser.add_argument('--valid_batch', help='Size of Batch for Validation, Test', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--save_every', help='', type=int, default=1)
    
    parser.add_argument('--save_dir', help='Full save directory', type=str, default='F:\Projects\OutfitTransformer\outfit_transformer\checkpoints')
    parser.add_argument('--data_dir', help='Full dataset directory', type=str, default='F:\Projects\datasets\polyvore_outfits')
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--scheduler_step_size', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-5)

    parser.add_argument('--wandb_api_key', default=None)
    parser.add_argument('--checkpoint', default=None)

    args = parser.parse_args()

    # Wandb
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        os.environ["WANDB_PROJECT"] = f"OutfitTransformer-{args.model}"
        os.environ["WANDB_LOG_MODEL"] = "all"
        wandb.login()
        run = wandb.init()

    # Setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.task == 'cp':
        dataset_type = 'cp'
    elif args.task == 'cir':
        dataset_type = 'outfit'

    train_dataset_args = DatasetArguments(
        polyvore_split = 'nondisjoint',
        task_type = dataset_type,
        dataset_type = 'train',
        outfit_max_length=args.outfit_max_length,
        use_text=True
        )
    valid_dataset_args = DatasetArguments(
        polyvore_split = 'nondisjoint',
        task_type = dataset_type,
        dataset_type = 'valid',
        outfit_max_length=args.outfit_max_length,
        use_text=True
        )
    valid_fitb_dataset_args = DatasetArguments(
        polyvore_split = 'nondisjoint',
        task_type = 'fitb',
        dataset_type = 'valid',
        outfit_max_length=12,
        use_text=True
        )
    valid_cp_dataset_args = DatasetArguments(
        polyvore_split = 'nondisjoint',
        task_type = 'cp',
        dataset_type = 'valid',
        outfit_max_length=12,
        use_text=True
        )
    
    HUGGING_FACE = 'sentence-transformers/paraphrase-albert-small-v2'
    tokenizer = AutoTokenizer.from_pretrained(HUGGING_FACE)

    train_dataloader = DataLoader(PolyvoreDataset(args.data_dir, train_dataset_args, tokenizer),
                                args.train_batch, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(PolyvoreDataset(args.data_dir, valid_dataset_args, tokenizer),
                                args.valid_batch, shuffle=False, num_workers=args.num_workers)
    valid_cp_dataloader = DataLoader(PolyvoreDataset(args.data_dir, valid_cp_dataset_args, tokenizer), 
                                       args.valid_batch, shuffle=False, num_workers=args.num_workers)
    valid_fitb_dataloader = DataLoader(PolyvoreDataset(args.data_dir, valid_fitb_dataset_args, tokenizer), 
                                       args.valid_batch, shuffle=False, num_workers=args.num_workers)

    model = OutfitTransformer(
        embedding_dim=args.embedding_dim,
        txt_huggingface = HUGGING_FACE
        )
    print('[COMPLETE] Build Model')
    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f'[COMPLETE] Load Model from {checkpoint}')

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.5)

    model.fit(
        task = args.task,
        save_dir = args.save_dir,
        n_epochs = args.n_epochs,
        optimizer = optimizer,
        train_dataloader = train_dataloader,
        valid_dataloader = valid_dataloader,
        valid_cp_dataloader = valid_cp_dataloader,
        valid_fitb_dataloader = valid_fitb_dataloader,
        valid_cir_dataloader = None,
        device = 'cuda',
        use_wandb = True if args.wandb_api_key else False,
        save_every = args.save_every,
        )