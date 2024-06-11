# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import os
import math
import datetime
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.utils.utils import *
from src.datasets.processor import *
import pickle


@dataclass
class DatasetArguments:
    polyvore_split: str = 'nondisjoint'
    task_type: str = 'cp'
    dataset_type: str = 'train'
    

class PolyvoreDataset(Dataset):

    def __init__(
            self,
            data_dir: str,
            args: DatasetArguments,
            input_processor: FashionInputProcessor
            ):
        # Directory settings
        self.img_dir = os.path.join(data_dir, 'images')
        # Arguments
        self.args = args
        # Meta Data preprocessing
        meta_data_path = os.path.join(data_dir, 'polyvore_item_metadata.json')
        self.data = list(json.load(open(meta_data_path)).items())

        self.input_processor = input_processor

    def _load_img(self, item_id):
        path = os.path.join(self.img_dir, f"{item_id}.jpg")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        item_id = self.data[idx][0]
        #
        category = '<' + self.data[idx][1]['semantic_category'] + '>'
        #
        image = self._load_img(item_id)
        #
        desc = self.data[idx][1]['title']
        if not desc:
            desc = self.data[idx][1]['url_name']
        text = desc.replace('\n','').strip().lower()
        #
        inputs = self.input_processor.preprocess(category, image, text)
        return item_id, inputs
        
    def __len__(self):
        return len(self.data)