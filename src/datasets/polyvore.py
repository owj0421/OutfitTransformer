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

from PIL import Image


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
            input_processor: FashionInputProcessor,
            ):
        # Arguments
        self.args = args
        self.is_train = (args.dataset_type == 'train')
        # Meta Data preprocessing
        self.item_ids, self.item_id2idx, self.item_id2category, self.category2item_ids, \
            self.categories, self.outfit_id2item_id, self.item_id2desc = load_data(data_dir, args)
        
        self.img_dir = os.path.join(data_dir, 'images')
        self.input_processor = input_processor
            
        # Input Type
        if args.task_type == 'cp':
            self.data = load_cp_inputs(data_dir, args, self.outfit_id2item_id)
        elif args.task_type == 'fitb':
            self.data = load_fitb_inputs(data_dir, args, self.outfit_id2item_id)
        elif args.task_type == 'cir':
            self.data = load_triplet_inputs(data_dir, args, self.outfit_id2item_id)
        else:
            raise ValueError('task_type must be one of "cp", "fitb", and "cir".')

    def _load_img(self, item_id):
        path = os.path.join(self.img_dir, f"{item_id}.jpg")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_txt(self, item_id):
        desc = self.item_id2desc[item_id] if item_id in self.item_id2desc else self.item_id2category[item_id]

        return desc
    
    def _get_inputs(self, item_ids: List[int], pad: bool=False) -> Dict[str, Tensor]:
        category = [self.item_id2category[item_id] for item_id in item_ids]
        images = [self._load_img(item_id) for item_id in item_ids]
        texts = [self._load_txt(item_id) for item_id in item_ids]
        
        return self.input_processor(category, images, texts=texts, do_pad=pad)

    def __getitem__(self, idx):
        if self.args.task_type == 'cp':
            targets, outfit_ids = self.data[idx]
            inputs = self._get_inputs(outfit_ids, pad=True)

            return {'targets': targets, 'inputs': inputs}
        
        elif self.args.task_type =='fitb':
            question_ids, candidate_ids = self.data[idx]
            questions = self._get_inputs(question_ids, pad=True)
            candidates = self._get_inputs(candidate_ids)

            return  {'questions': questions, 'candidates': candidates} # ans is always 0 index

        elif self.args.task_type =='cir':
            item_ids = self.data[idx]
            np.random.shuffle(item_ids)
            
            anchor_item_ids = item_ids[1:]
            positive_item_ids = [item_ids[0]]
            
            anchor = self._get_inputs(anchor_item_ids, pad=True)
            positive = self._get_inputs(positive_item_ids)

            return {'anchor': anchor, 'positive': positive}
        
    def __len__(self):
        return len(self.data)
    

def load_fitb_inputs(data_dir, args, outfit_id2item_id):
    fitb_path = os.path.join(data_dir, args.polyvore_split, f'fill_in_blank_{args.dataset_type}.json')
    with open(fitb_path, 'r') as f:
        fitb_data = json.load(f)
        fitb_inputs = []
        for item in fitb_data:
            question_ids = list(map(lambda x: outfit_id2item_id[x], item['question']))
            candidate_ids = list(map(lambda x: outfit_id2item_id[x], item['answers']))
            fitb_inputs.append((question_ids, candidate_ids))
    return fitb_inputs


def load_cp_inputs(data_dir, args, outfit_id2item_id):
    cp_path = os.path.join(data_dir, args.polyvore_split, f'compatibility_{args.dataset_type}.txt')
    cp_inputs = []
    with open(cp_path, 'r') as f:
        cp_data = f.readlines()
        for d in cp_data:
            target, *item_ids = d.split()
            cp_inputs.append((torch.FloatTensor([int(target)]), list(map(lambda x: outfit_id2item_id[x], item_ids))))
    return cp_inputs


def load_triplet_inputs(data_dir, args, outfit_id2item_id):
    outfit_data_path = os.path.join(data_dir, args.polyvore_split, f'{args.dataset_type}.json')
    outfit_data = json.load(open(outfit_data_path))
    triplet_inputs = [[outfit['items'][i]['item_id'] 
                       for i in range(len(outfit['items']))] 
                       for outfit in outfit_data]
    triplet_inputs = list(filter(lambda x: len(x) > 1, triplet_inputs))
    return triplet_inputs

def load_data(data_dir, args):
    # Paths
    # data_dir = os.path.join(data_dir, args.polyvore_split)
    outfit_data_path = os.path.join(data_dir, args.polyvore_split, f'{args.dataset_type}.json')
    meta_data_path = os.path.join(data_dir, 'polyvore_item_metadata.json')
    outfit_data = json.load(open(outfit_data_path))
    meta_data = json.load(open(meta_data_path))
    # Load
    item_ids = set()
    categories = set()
    item_id2category = {}
    item_id2desc = {}
    category2item_ids = {}
    outfit_id2item_id = {}
    for outfit in outfit_data:
        outfit_id = outfit['set_id']
        for item in outfit['items']:
            # Item of cloth
            item_id = item['item_id']
            item_ids.add(item_id)
            # Category of cloth
            category = '<' + meta_data[item_id]['semantic_category'] + '>'
            categories.add(category)
            item_id2category[item_id] = category
            if category not in category2item_ids:
                category2item_ids[category] = set()
            category2item_ids[category].add(item_id)
            # Description of cloth
            desc = meta_data[item_id]['title']
            if not desc:
                desc = meta_data[item_id]['url_name']
            item_id2desc[item_id] = desc.replace('\n','').strip().lower()
            # Replace the item code with the outfit number with the image code
            outfit_id2item_id[f"{outfit['set_id']}_{item['index']}"] = item_id
            
    item_ids = list(item_ids)
    item_id2idx = {id : idx for idx, id in enumerate(item_ids)}
    categories = list(categories)

    return item_ids, item_id2idx, \
        item_id2category, category2item_ids, categories, \
            outfit_id2item_id, item_id2desc