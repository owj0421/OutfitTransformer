import os
import math
import wandb
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.utils import *
from src.loss.focal_loss import focal_loss
from src.loss.triplet_loss import triplet_loss
from src.loss.info_nce import InfoNCE
from src.models.recommender import RecommendationModel

def get_masked_indices(model: RecommendationModel, embedding_model_outputs: Dict[str, Any], mom_probability: float=0.3):
    token_ids = embedding_model_outputs['category_ids']
    special_token_cnt = len(model.input_processor.special_tokens)

    probability_matrix = torch.full(embedding_model_outputs['mask'].shape, mom_probability)
    special_token_mask = (token_ids < special_token_cnt).to(probability_matrix.device)
    probability_matrix.masked_fill_(special_token_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    return masked_indices

def masked_outfit_modeling(inputs, model: RecommendationModel, criterion, mom_probability: float=0.3):
    embedding_model_outputs = model.encode(inputs)

    masked_indices = get_masked_indices(model, embedding_model_outputs, mom_probability)

    y_true = embedding_model_outputs['embeds'][masked_indices]
    device = y_true.get_device() if y_true.device.type == 'cuda' else torch.device('cpu')
    # 마스크 임베딩 꺼낸 후
    mask_embedding = model.category_embeddings(torch.LongTensor([1]).to(device)).squeeze()
    # 마스킹 할 자리를 0으로 만든 후, 마스크 임베딩을 더해준다.
    embedding_model_outputs['embeds'] = embedding_model_outputs['embeds'].masked_fill_(masked_indices[..., None].to(device), value=0.0)
    embedding_model_outputs['embeds'][masked_indices] = embedding_model_outputs['embeds'][masked_indices] + mask_embedding

    last_hidden_state = model(embedding_model_outputs)
    y_pred = last_hidden_state[masked_indices]

    loss = criterion(y_pred, y_true)
    return loss