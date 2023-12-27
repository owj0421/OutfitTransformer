import os
import os.path
from torch.utils.data import Dataset
import numpy as np
import random
import json
import torch
from transformers import AutoTokenizer
from dataclasses import dataclass
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2



import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.model import *
from model.encoder import ItemEncoder


class Preprocessor():
    def __init__(self, img_size=224):
        huggingface_tokenizer ='sentence-transformers/paraphrase-albert-small-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_tokenizer)
        self.max_token_len = 16
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2()
            ])
        
    def _preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        return image

    def _preprocess_desc(self, desc):
        input_ids, _, attention_mask, *_  = self.tokenizer(desc, max_length=self.max_token_len, padding='max_length', truncation=True, return_tensors='pt').values()
        return input_ids.squeeze(0), attention_mask.squeeze(0)

    def _preprocess_item(self, image_path, desc):
        image = self._preprocess_image(image_path)
        input_ids, attention_mask = self._preprocess_desc(desc)
        return image, input_ids, attention_mask

    def preprocess_outfit(self, outfit):
        image_batch = []
        input_ids_batch = []
        attention_mask_batch = []

        for category, info in outfit.items():
            image_path = info.get('image_path')
            desc = info.get('desc', category)

            image, input_ids, attention_mask = self._preprocess_item(image_path, desc)

            image_batch.append(image)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)

        image_batch = torch.stack(image_batch)
        input_ids_batch = torch.stack(input_ids_batch)
        attention_mask_batch = torch.stack(attention_mask_batch)

        return image_batch, input_ids_batch, attention_mask_batch


preprocessor = Preprocessor()

device = torch.device('cuda')
checkpoint = torch.load('F:/Projects/outfit-transformer/checkpoints/compatibility/2023-12-26/7_0.902.pth')

model = OutfitTransformer(embedding_dim=128).to(device)
encoder = ItemEncoder(embedding_dim=128).to(device)


model.load_state_dict(checkpoint['model_state_dict'], strict=False)
encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)

model.eval()
encoder.eval()

###
outfit = {
    'top': {
        'image_path': 'F:/Projects/outfit-transformer/data/query_img.jpg',
        'desc': 'green formal shirt'
        },
    'bottom': {
        'image_path': 'F:/Projects/outfit-transformer/data/query_img.jpg',
        'desc': 'white formal trousers'
        },
    'shoes': {
        'image_path': 'F:/Projects/outfit-transformer/data/query_img.jpg',
        'desc': 'black formal shoes'
        },
    }

image_batch, input_ids_batch, attention_mask_batch = preprocessor.preprocess_outfit(outfit)

embed = encoder(image_batch.to(device), input_ids_batch.to(device), attention_mask_batch.to(device))
prob = model('compatibility',  embed.unsqueeze(0))
print(prob)