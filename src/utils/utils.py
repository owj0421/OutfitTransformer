# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import os
import numpy as np
import random
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from tqdm import tqdm


def stack_tensors(mask, tensor):
    B, S = mask.shape
    mask = mask.reshape(-1)
    s = list(tensor.shape)
    tensor = tensor.reshape([s[0] * s[1]] + s[2:])
    tensor = tensor[~mask]
    return tensor


def unstack_tensors(mask, tensor):
    B, S = mask.shape
    mask = mask.reshape(-1)
    new_shape = [B * S] + list(tensor.shape)[1:]
    device = tensor.device if tensor.device.type == 'cuda' else torch.device('cpu')
    new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=device)
    new_tensor[~mask] = tensor
    new_tensor = new_tensor.reshape([B, S] + list(tensor.shape)[1:])
    return new_tensor


def stack_dict(batch):
    mask = batch.get('mask')  # 'mask' 키가 없을 경우 None을 반환
    stacked_batch = {key: stack_tensors(mask, value) if (key != 'mask' and value != None) else value for key, value in batch.items()}
    return stacked_batch


def unstack_dict(batch):
    mask = batch.get('mask')  # 'mask' 키가 없을 경우 None을 반환
    unstacked_batch = {key: unstack_tensors(mask, value) if (key != 'mask' and value != None) else value for key, value in batch.items()}
    return unstacked_batch


def unstack_output(output):
    for i in output.__dict__.keys():
        if i == 'mask':
            continue
        attr = getattr(output, i)
        if isinstance(attr, list):
            setattr(output, i, [unstack_tensors(output.mask, i) for i in attr])
        elif isinstance(attr, Tensor):
            setattr(output, i, unstack_tensors(output.mask, attr))
    return output


def save_model(model, save_dir, model_name, device):
    
    def _create_folder(dir):
        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
        except OSError:
            print('[Error] Creating directory.' + dir)

    _create_folder(save_dir)

    model.cpu()
    path = os.path.join(save_dir, f'{model_name}.pth')
    checkpoint = {
        'state_dict': model.state_dict()
        }
    torch.save(checkpoint, path)
    print(f'[COMPLETE] Save at {path}')
    model.to(device)