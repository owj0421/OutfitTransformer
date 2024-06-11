# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from functools import reduce

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from dataclasses import dataclass
from transformers import CLIPImageProcessor, CLIPTokenizer

class FashionInputProcessor:
    """Processor that pre-processes input in the form of DeepFashion model inputs.
    The DeepFashion model must receive the output from this processor as an input.

    :param categories       : list of all categories. 
    :param use_image        : Sent True if you want to use image.
    :param image_processor  : (If using images) DeepFashionImageProcessor instance specifying image preprocessing (transformation, etc.).
    :param use_text         : Sent True if you want to use text. Only at most one of use_text and use_text_feature must be true.
    :param text_tokenizer   : (If using texts) Huggingface tokenizer instance.
    :param text_max_length  : (If using texts) Maximum token length to tokenize.
    :param text_padding     : (If using texts) True will pad text to the maximum length.
    :param text_truncation  : (If using texts) True will truncated text to the maximum length.
    :param use_text_feature : Sent True if you want to use text feature directly. Only at most one of use_text and use_text_feature must be true.
    :param text_feature_dim : (If using text features) Text feature dimension.
    :param outfit_max_length: Maximum number of clothes. Input will truncated based on this value.
    :return: A Processor instance.
    """
    def __init__(
            self,
            categories: Optional[List[str]] = None, 
            use_image: bool = True,
            image_processor: Optional[Any] = None, 
            use_text: bool = False,
            text_tokenizer: Optional[AutoTokenizer] = None, 
            text_max_length: int = 32, 
            text_padding: str = 'max_length', 
            text_truncation: bool = True, 
            outfit_max_length: int = 16
            ):
        if categories is None:
            raise ValueError("You need to specify a `categories`.")
        # if (~use_image and ~use_text and ~use_text_feature):
        #     raise ValueError("At least one of `use_image`, `use_text` and `use_text_feature` should be set as `True.`")
        if use_image and image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if use_text and text_tokenizer is None:
            raise ValueError("If you want to set `use_text` as `True`, You need to specify a `text_tokenizer`.")

        categories = reduce(lambda acc, cur: acc if cur in acc else acc+[cur], categories, [])

        self.category_special_tokens = ['<pad>', '<mask>']
        self.category_tokens = categories
        
        self.id2token = self.category_special_tokens + self.category_tokens
        self.token2id = {v: i for i, v, in enumerate(self.id2token)}

        self.use_image = use_image
        self.image_processor = image_processor

        self.use_text = use_text
        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length
        self.text_padding = text_padding
        self.text_truncation = text_truncation
        
        self.outfit_max_length = outfit_max_length

        # Items for pad
        self.CATEGORY_PAD = '<pad>'
        if use_text:
            self.TEXT_PAD = text_tokenizer.pad_token
        if use_image:
            if isinstance(self.image_processor, CLIPImageProcessor):
                self.IMAGE_PAD = np.zeros((3, image_processor.size['shortest_edge'], image_processor.size['shortest_edge']), dtype='uint8') # CLIP 3, 244, 244
            else:
                self.IMAGE_PAD = np.zeros((3, image_processor.size, image_processor.size), dtype='uint8')

    def __call__(
            self, 
            category: Optional[List[str]]=None, 
            images: Optional[List[ndarray]]=None, 
            texts: Optional[List[str]]=None, 
            do_pad: bool=True,
            **kwargs
            ) -> Dict[str, Tensor]:
        return self.preprocess_batch(category, images, texts, do_pad)

    def preprocess(self, category=None, image=None, text=None, **kwargs):
        outputs = {'mask': None, 'category_ids': None, 'image_features': None, 'input_ids': None, 'attention_mask': None}

        outputs['mask'] = torch.BoolTensor([False])
        
        if category is not None:
            category_ids = torch.LongTensor([self.token2id[category]])
            outputs['category_ids'] = category_ids
            
        if image is not None:
            image_features = self.image_processor(image, **kwargs)['pixel_values'][0]
            if isinstance(image_features, np.ndarray):
                image_features = torch.from_numpy(image_features)
            outputs['image_features'] = image_features
            
        if text is not None:
            text_ = self.text_tokenizer([text], max_length=self.text_max_length, padding=self.text_padding, truncation=self.text_truncation, return_tensors='pt')
            outputs['input_ids'] = text_['input_ids'].squeeze(0)
            outputs['attention_mask'] = text_['attention_mask'].squeeze(0)

        return outputs

    def preprocess_batch(
            self, 
            categories: Optional[List[str]]=None, 
            images: Optional[List[ndarray]]=None, 
            texts: Optional[List[str]]=None, 
            do_pad: bool=True,
            **kwargs
            ) -> Dict[str, Tensor]:
        
        get_item_num = lambda x: len(x) if x else 0
        num_items = min(self.outfit_max_length, max([get_item_num(x) for x in [categories, images, texts]]))
        
        inputs = {'mask': [], 'category_ids': [], 'image_features': [], 'input_ids': [], 'attention_mask': []}
        for item_idx in range(num_items):
            category = categories[item_idx] if categories else None
            image = images[item_idx] if images else None
            text = texts[item_idx] if texts else None
            for k, v in self.preprocess(category, image, text).items():
                if v is None:
                    continue
                inputs[k].append(v)

        for k in list(inputs.keys()):
            if len(inputs[k]) > 0:
                if do_pad:
                    pad_item = torch.zeros_like(inputs[k][-1]) if k != 'mask' else torch.BoolTensor([True])
                    inputs[k] += [pad_item for _ in range(self.outfit_max_length - num_items)]
                inputs[k] = torch.stack(inputs[k])
            else:
                del inputs[k]
                
        inputs['mask'] = inputs['mask'].squeeze(1)
        
        return inputs
    

class FashionImageProcessor:
    """Processor that pre-processes image inputs.

    :param size                : Size to resize images. Both horizontal and vertical are the same.
    :param use_custom_transform: True will add `custom_transfrom`.
    :param use_custom_transform: (If using custom transform) A list of Albumentation transforms.

    :return: A ImageProcessor instance.
    """
    def __init__(
            self,
            size: int = 224,
            use_custom_transform: bool = False,
            custom_transform: Optional[List[Any]] = None
            ):
        self.size = size
        transform = list()

        if use_custom_transform:
            transform += custom_transform

        if not any(isinstance(x, A.Resize) for x in transform):
            transform.append(A.Resize(size, size))

        if not any(isinstance(x, A.Normalize) for x in transform):
            transform.append(A.Normalize())

        if not any(isinstance(x, ToTensorV2) for x in transform):
            transform.append(ToTensorV2())

        self.transform = A.Compose(transform)

    def __call__(
            self,
            images: List[ndarray]
            ) -> Tensor:
        """
        :param images       : List of images of cloths, which is first converted to ndarray via cv2.
        :return: Preprocessed image as Tensor form.
        """

        if len(images.shape) > 3:
            images = [self.transform(image=image)['image'] for image in images]
        else:
            images = [self.transform(image=images)['image']]

        return {'pixel_values': images}
    


