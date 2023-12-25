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

@dataclass
class DatasetArgs():
    work_dir: str
    data_dir: str
    polyvore_split: str='nondisjoint'
    max_token_len: int=16
    img_size: int=224
    huggingface_tokenizer: str='sentence-transformers/paraphrase-albert-small-v2'


class PolyvoreDataset(Dataset):
    def __init__(
            self,
            args: DatasetArgs,
            task: str='compatibility',
            dataset_type: str='train',
            train_transform=None,
            ):
        # Dataset configurations
        self.task = task
        self.is_train = (dataset_type == 'train')
        self.dataset_type = dataset_type # train, test, val
        self.img_size = args.img_size
        self.max_token_len = args.max_token_len

        # Path configurations
        self.data_dir = os.path.join(args.data_dir, args.polyvore_split)
        self.img_dir = os.path.join(args.data_dir, 'images')
        self.meta_data_path = os.path.join(args.data_dir, 'polyvore_item_metadata.json')
        self.outfit_data_path = os.path.join(self.data_dir, f'{dataset_type}.json')
        self.fitb_path = os.path.join(self.data_dir,  f'fill_in_blank_{self.dataset_type}.json')
        self.compatibility_path = os.path.join(self.data_dir, f'compatibility_{self.dataset_type}.txt')
        
        # Image Data Configurations
        self.train_transform = train_transform
        self.transform = A.Compose([
            A.Resize(args.img_size, args.img_size),
            A.Normalize(),
            ToTensorV2()
            ])
        
        # Text data configurations
        self.tokenizer = AutoTokenizer.from_pretrained(args.huggingface_tokenizer)

        # Data preprocessing
        outfit_data = json.load(open(self.outfit_data_path))
        meta_data = json.load(open(self.meta_data_path))

        item_ids = set()
        max_set_len = 0 # Which means the length of the outfit that contains the most clothes
        item_id2category = {}
        item_id2desc = {}
        category2item_ids = {}
        outfit_id2item_id = {}
        for outfit in outfit_data:
            outfit_id = outfit['set_id']
            max_set_len = max(max_set_len, len(outfit['items']))
            for item in outfit['items']:
                # Item of cloth
                item_id = item['item_id']
                # Category of cloth
                category = meta_data[item_id]['semantic_category']
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
                item_ids.add(item_id)
        item_ids = list(item_ids)

        # Data
        self.item_ids = item_ids
        self.item_id2idx = {id : idx for idx, id in enumerate(item_ids)}
        self.max_set_len = max_set_len
        self.item_id2category = item_id2category
        self.item_id2desc = item_id2desc
        self.category2item_ids = category2item_ids
        self.outfit_id2item_id = outfit_id2item_id

        # Pseudo item to pad outfit sequence
        self.pad_item = [torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32), 'PAD']
        
        # Query image for Transformer input
        query_img = cv2.imread(os.path.join(args.work_dir, 'data', 'query_img.jpg'))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        query_img = self.transform(image=query_img)['image']
        self.query_img = query_img

        # For Pretrain
        # Compatibility Question
        if self.task == 'compatibility':
            with open(self.compatibility_path, 'r') as f:
                compatibility_data = f.readlines()
                self.data = []
                for d in compatibility_data:
                    y, *items = d.split()
                    self.data.append((int(y), list(map(lambda x: self.outfit_id2item_id[x], items))))
        # For fine-tuning
        else:
            # Complete outfit combination
            if self.is_train:
                self.data = [[outfit['items'][i]['item_id'] for i in range(len(outfit['items']))] for outfit in outfit_data]
                self.data = list(filter(lambda x: len(x) > 1, self.data))
            # FITB Question for test ans validation splits
            else:
                with open(self.fitb_path, 'r') as f:
                    fitb_data = json.load(f)
                    questions = []
                    for item in fitb_data:
                        question_ids = list(map(lambda x: self.outfit_id2item_id[x], item['question']))
                        candidate_ids = list(map(lambda x: self.outfit_id2item_id[x], item['answers']))
                        ans_idx = 0
                        questions.append((question_ids, candidate_ids, ans_idx))
                self.data = questions

    def _load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.is_train and self.train_transform:
            img = self.train_transform(image=img)['image']
        else:
            img = self.transform(image=img)['image']
        return img
    
    def _get_item(self, item_id):
        img_path = os.path.join(self.img_dir, f"{item_id}.jpg")
        img = self._load_img(img_path)

        if item_id in self.item_id2desc:
            desc = self.item_id2desc[item_id]
        else:
            desc = "cloth"
        return img, desc

    def _preprocess_desc(self, desc):
        input_ids, _, attention_mask, *_  = self.tokenizer(desc, max_length=self.max_token_len, padding='max_length', truncation=True, return_tensors='pt').values()
        return input_ids, attention_mask

    def _preprocess_items(self, set_, pad=False):
        if pad:
            set_mask = torch.ones((self.max_set_len), dtype=torch.long)
            set_mask[:len(set_)] = 0
            set_ = set_ + [self.pad_item for _ in range(self.max_set_len - len(set_))]
        img, desc = zip(*set_)
        img = torch.stack(img)
        input_ids, attention_mask = self._preprocess_desc(desc)

        return  set_mask.bool() if pad else None, img, input_ids, attention_mask
    
    def _sample_neg(self, pos_id, n, same_category=False, ignore_ids=None):
        if same_category:
            pos_category = self.item_id2category[pos_id]
            pool = self.category2item_ids[pos_category] - set([pos_id])
            return  random.sample(pool, n)
        else:
            pool = set(self.item_ids) - set(ignore_ids)
            return  random.sample(pool, n)

    def __getitem__(self, idx):
        if self.task == 'compatibility':
            y, inputs = self.data[idx]
            y = torch.FloatTensor([y])
            inputs = [self._get_item(i) for i in inputs]
            set_mask, img, input_ids, attention_mask = self._preprocess_items(inputs, pad=True)

            return y, set_mask, img, input_ids, attention_mask

        else:
            if self.is_train:
                anchor_ids = self.data[idx]
                anchor_items = [self._get_item(item_id) for item_id in anchor_ids]

                pos_idx = np.random.choice(range(len(anchor_ids)))
                pos_id = anchor_ids[pos_idx]
                pos_item = [anchor_items[pos_idx]]
                del anchor_items[pos_idx]

                query_item = (self.query_img, self.item_id2category[pos_id])
                input_items = [query_item] + anchor_items # Transformer input

                hard_neg_ids = self._sample_neg(pos_id=pos_id, n=6, same_category=True, ignore_ids=anchor_ids)
                soft_neg_ids = self._sample_neg(pos_id=pos_id, n=4, same_category=False, ignore_ids=anchor_ids)
                neg_ids = hard_neg_ids + soft_neg_ids
                
                neg_items = [self._get_item(neg_id) for neg_id in neg_ids]
            
                input_set_mask, input_img, input_input_ids, input_attention_mask = self._preprocess_items(input_items, pad=True)
                _, pos_img, pos_input_ids, pos_attention_mask = self._preprocess_items(pos_item, pad=False)
                _, neg_img, neg_input_ids, neg_attention_mask = self._preprocess_items(neg_items, pad=False)
                
                return input_set_mask, input_img, input_input_ids, input_attention_mask,\
                    pos_img, pos_input_ids, pos_attention_mask,\
                    neg_img, neg_input_ids, neg_attention_mask

            else:
                question_ids, candidate_ids, ans_idx = self.data[idx]

                question_items = [self._get_item(item_id) for item_id in question_ids]
                query_item = (self.query_img, self.item_id2category[candidate_ids[ans_idx]])
                question_items = [query_item] + question_items

                candidate_items = [self._get_item(item_id) for item_id in candidate_ids]

                question_set_mask, question_img, question_input_ids, question_attention_mask = self._preprocess_items(question_items, pad=True)
                _, candidate_img, candidate_input_ids, candidate_attention_mask = self._preprocess_items(candidate_items, pad=False)
                ans_idx = torch.FloatTensor([ans_idx])

                return  question_set_mask, question_img, question_input_ids, question_attention_mask,\
                    candidate_img, candidate_input_ids, candidate_attention_mask,\
                    ans_idx

    def __len__(self):
        return len(self.data)