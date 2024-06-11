# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import os
import math
import wandb
from tqdm import tqdm
from itertools import chain
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def safe_divide(a, b, eps=1e-7):
    return a / (b + eps)

def triplet_loss(
          anchor_embeds,
          positive_embeds, 
          margin: float = 2,
          method: Literal['batch_all', 'batch_hard'] = 'batch_hard',
          aggregation: Literal['mean', 'sum', 'none'] = 'mean'
          ) -> Tensor:
        """Triplet margin loss.
        Implemented in Online mining manner.

        :param outfit_outs: Should instance of DeepFashionOutput
        :param margin: Margin for Triplet loss.
        :param method: How to handle a negative sample.
            - 'batch_all': Use the average of all negatives in the mini-batch.
            - 'batch_hard': Use the most hardest sample in the mini-batch.
        :param aggregation: How to handle a final output. same as torch triplet loss' one.
        :return: Triplet margin loss
        """
        device = anchor_embeds.get_device()

        dist_matrix = nn.PairwiseDistance()(anchor_embeds.unsqueeze(1), positive_embeds.unsqueeze(0))

        # Compute Positive distances
        # dist_matrix의 대각 요소를 취한다.
        pos_dist = dist_matrix.diag()

        # Compute Negative distances
        # dist_matrix의 대각 요소를 0
        neg_matrix = dist_matrix * torch.ones_like(dist_matrix).fill_diagonal_(0.)


        if method == 'both':
            neg_dist_all = safe_divide(torch.sum(neg_matrix, dim=-1), torch.count_nonzero(neg_matrix, dim=-1))
            hinge_dist_all = torch.clamp(margin + pos_dist - neg_dist_all, min=0.0)
            loss_all = torch.mean(hinge_dist_all)

            neg_matrix[neg_matrix < pos_dist.expand_as(neg_matrix)] = 0.
            neg_matrix = neg_matrix + torch.where(neg_matrix == 0., torch.max(neg_matrix), 0.)
            neg_dist_hard, _ = torch.min(neg_matrix, dim=-1, keepdim=True)
            hinge_dist_hard = torch.clamp(margin + pos_dist - neg_dist_hard, min=0.0)
            loss_hard = torch.mean(hinge_dist_hard)

            return loss_all + loss_hard
        
        if method == 'batch_all':
            neg_dist = safe_divide(torch.sum(neg_matrix, dim=-1), torch.count_nonzero(neg_matrix, dim=-1))
        elif method == 'batch_hard':
            neg_matrix[neg_matrix < pos_dist.expand_as(neg_matrix)] = 0.
            neg_matrix = neg_matrix + torch.where(neg_matrix == 0., torch.max(neg_matrix), 0.)
            neg_dist, _ = torch.min(neg_matrix, dim=-1, keepdim=True)
        else:
            raise ValueError('task_type must be one of `batch_all` and `batch_hard`.')
        
        hinge_dist = torch.clamp(margin + pos_dist - neg_dist, min=0.0)
        if aggregation == 'mean':
            loss = torch.mean(hinge_dist)
        elif aggregation == 'sum':
            loss = torch.sum(hinge_dist)
        elif aggregation == 'none':
            loss = hinge_dist
        else:
            raise ValueError('aggregation must be one of `mean`, `sum` and `none`.')
        
        return loss