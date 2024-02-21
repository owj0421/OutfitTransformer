# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from functools import reduce

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer


class DeepFashionImageProcessor:
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
        images = [self.transform(image=image)['image'] for image in images]

        return torch.stack(images)


class DeepFashionInputProcessor:
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
            image_processor: Optional[DeepFashionImageProcessor] = None, 
            use_text: bool = False,
            text_tokenizer: Optional[AutoTokenizer] = None, 
            text_max_length: int = 8, 
            text_padding: str = 'max_length', 
            text_truncation: bool = True, 
            use_text_feature: bool = False,
            text_feature_dim: Optional[int] = None,
            outfit_max_length: int = 8
            ):
        if categories is None:
            raise ValueError("You need to specify a `categories`.")
        if ~(~use_image and ~use_text and ~use_text_feature):
            raise ValueError("At least one of `use_image`, `use_text` and `use_text_feature` should be set as `True.`")
        if use_image and image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if use_text and use_text_feature is None:
            raise ValueError("Only at most one of `use_text` and `use_text_feature` can be set as `True.`")
        if use_text and text_tokenizer is None:
            raise ValueError("If you want to set `use_text` as `True`, You need to specify a `text_tokenizer`.")
        if use_text_feature and text_feature_dim is None:
            raise ValueError("If you want to set `use_text_feature` as `True`, You need to specify a `text_feature_dim`.")
        
        categories = reduce(lambda acc, cur: acc if cur in acc else acc+[cur], categories, [])
        if 'pad' not in categories:
            categories.append('pad')
        self.categories = categories

        self.category_encoder = LabelEncoder()
        self.category_encoder.fit(self.categories)

        self.use_image = use_image
        self.image_processor = image_processor

        self.use_text = use_text
        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_max_length
        self.text_padding = text_padding
        self.text_truncation = text_truncation

        self.use_text_feature = use_text_feature
        self.text_feature_dim = text_feature_dim
        
        self.outfit_max_length = outfit_max_length

        # Items for pad
        self.CATEGORY_PAD = 'pad'
        if use_image:
            self.IMAGE_PAD = np.zeros((image_processor.size, image_processor.size, 3), dtype='uint8')
        if use_text:
            self.TEXT_PAD = '<PAD>'
        if use_text_feature:
            self.TEXT_FEATURE_PAD = np.zeros_like(text_feature_dim)

    def __call__(
            self, 
            category: Optional[List[str]]=None, 
            images: Optional[List[ndarray]]=None, 
            texts: Optional[List[str]]=None, 
            text_features: Optional[List[ndarray]]=None,
            do_pad: bool=False,
            do_truncation: bool=True,
            **kwargs
            ) -> Dict[str, Tensor]:
        """Preprocess set of outfit items.

        :param category     : List of category of cloths
        :param images       : (If using text images) List of images of cloths, which is first converted to ndarray via cv2.
        :param texts        : (If using texts) List of descriptions of cloths.
        :param text_features: (If using text features) List of text vectors of cloths
        :param do_pad       : True will pad item squence based on `self.outfit_max_lenght`.
        :param do_truncation: True will trancate item squence based on `self.outfit_max_lenght`.
        :return: A dictionary composed of a Deep Fashion model input.
        """
        if category is None:
            raise ValueError("You have to specify `category`.")
        if texts is None and images is None:
            raise ValueError("You have to specify either `texts` or `images`.")
        
        inputs = dict()

        if do_pad:
            mask = torch.ones((self.outfit_max_length), dtype=torch.bool)
            mask[:len(category)] = False
        else:
            mask = torch.zeros(min((len(category), self.outfit_max_length)), dtype=torch.bool)
        inputs['mask'] = mask
        
        if category is not None:
            if do_truncation:
                category = category[:self.outfit_max_length]
            if do_pad:
                category = category + [self.CATEGORY_PAD for _ in range(self.outfit_max_length - len(category))]
            inputs['category'] = torch.LongTensor(self.category_encoder.transform(category))
        
        if self.use_text_feature and text_features is not None:
            if do_truncation:
                text_features = text_features[:self.outfit_max_length]
            if do_pad:
                text_features = text_features + [self.TEXT_FEATURE_PAD for _ in range(self.outfit_max_length - len(text_features))]
            inputs['text_features'] = torch.stack([text_feature for text_feature in text_features])
        
        if self.use_text and texts is not None:
            if do_truncation:
                texts = texts[:self.outfit_max_length]
            if do_pad:
                texts = texts + [self.TEXT_PAD for _ in range(self.outfit_max_length - len(texts))]
            encoding = self.text_tokenizer(texts, max_length=self.text_max_length, padding=self.text_padding, truncation=self.text_truncation, return_tensors='pt')
            inputs['input_ids'] = encoding.input_ids
            inputs['attention_mask'] = encoding.attention_mask

        if self.use_image and images is not None:
            if do_truncation:
                images = images[:self.outfit_max_length]
            if do_pad:
                images = images + [self.IMAGE_PAD for _ in range(self.outfit_max_length - len(images))]
            inputs['image_features'] = self.image_processor(images, **kwargs)

        return inputs