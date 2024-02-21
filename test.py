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
    parser.add_argument('--polyvore_split', help='', type=str, default='nondisjoint')
    parser.add_argument('--test_batch', help='Size of Batch for Validation, Test', type=int, default=32)
    parser.add_argument('--data_dir', help='Full dataset directory', type=str, default='F:\Projects\datasets\polyvore_outfits')
    parser.add_argument('--checkpoint', default=None)

    args = parser.parse_args()

    # Setup
    device = torch.device(0) if torch.cuda.is_available() else torch.device(1)

    HUGGING_FACE = 'sentence-transformers/paraphrase-albert-small-v2'
    model = OutfitTransformer(
        embedding_dim=args.embedding_dim,
        txt_huggingface = HUGGING_FACE
        )
    print('[COMPLETE] Build Model')
    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f'[COMPLETE] Load Model from {checkpoint}')

    tokenizer = AutoTokenizer.from_pretrained(HUGGING_FACE)

    
    if args.task=='cp':
        test_cp_dataset_args = DatasetArguments(
            polyvore_split = args.polyvore_split,
            task_type = 'cp',
            dataset_type = 'test',
            outfit_max_length=16,
            use_text=True
            )
        test_fitb_dataset_args = DatasetArguments(
            polyvore_split = args.polyvore_split,
            task_type = 'fitb',
            dataset_type = 'test',
            outfit_max_length=16,
            use_text=True
            )
        test_cp_dataloader = DataLoader(PolyvoreDataset(args.data_dir, test_cp_dataset_args, tokenizer), 
                                        args.test_batch, shuffle=False)
        test_fitb_dataloader = DataLoader(PolyvoreDataset(args.data_dir, test_fitb_dataset_args, tokenizer), 
                                        args.test_batch, shuffle=False)
    elif args.task=='cir':
        pass

    model.to(device)
    model.eval()
    with torch.no_grad():
        if args.task=='cp':
            cp_score = model.cp_evaluation(
                 dataloader = test_cp_dataloader,
                 epoch = 0,
                 is_test = True,
                 device = device,
                 )
            fitb_score = model.fitb_evaluation(
                dataloader = test_fitb_dataloader,
                epoch = 0,
                is_test = False,
                device = device
                )
            print(f'TEST | Task: {args.test_task} | CP(AUC): {cp_score:.5f} | FITB(Acc): {fitb_score:.5f}')

        elif args.task=='cir':
            pass
        