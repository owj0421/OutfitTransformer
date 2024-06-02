import torch
from torch.utils.data import DataLoader

from src.datasets.polyvore import DatasetArguments, PolyvoreDataset

from src.models.embedder import CLIPEmbeddingModel
from src.models.recommender import RecommendationModel
from src.models.load import load_model
from src.loss.focal_loss import focal_loss
from src.utils.utils import save_model

import os
import wandb
import numpy as np

from bitsandbytes.optim import AdamW8bit
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score

@ dataclass
class Args:
    pass


args = Args()
# Dir Settings
args.data_dir = 'F:/Projects/datasets/polyvore_outfits'
args.checkpoint_dir = 'F:/Projects/OutfitTransformer/checkpoints'
args.model_path = None
args.load_model = True if args.model_path is not None else False
# Dataset & Input Processor Settings
args.polyvore_split = 'nondisjoint'
args.categories = ['<bottoms>', '<outerwear>', '<tops>', '<scarves>', '<hats>', '<all-body>', '<accessories>', '<sunglasses>', '<shoes>', '<jewellery>', '<bags>']
args.outfit_max_length = 16
args.use_image = True
args.use_text = True
args.use_category = False
args.text_max_length = 16
# Embedder&Recommender Model Settings
args.use_clip_embedding = True
args.clip_huggingface = 'patrickjohncyh/fashion-clip'
args.huggingface = 'sentence-transformers/all-mpnet-base-v2'
args.hidden = 128
args.n_layers = 3
args.n_heads = 16

# Training Setting
args.n_epochs = 10
args.num_workers = 0
args.train_batch_size = 64
args.val_batch_size = 64
args.lr = 1e-4
args.wandb_key = 'fa37a3c4d1befcb0a7b9b4d33799c7bdbff1f81f'
args.use_wandb = True if args.wandb_key else False
args.with_cuda = True


if __name__ == '__main__':
    TASK = 'cp'

    # Wandb
    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = args.wandb_key
        os.environ["WANDB_PROJECT"] = f"OutfitTransformer-{TASK}"
        os.environ["WANDB_LOG_MODEL"] = "all"
        wandb.login()
        run = wandb.init()
    
    date_info = datetime.today().strftime("%y%m%d")
    save_dir = os.path.join(args.checkpoint_dir, TASK, date_info)

    cuda_condition = torch.cuda.is_available() and args.with_cuda
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    
    model, input_processor = load_model(args)
    image_processor = CLIPImageProcessor.from_pretrained(args.clip_huggingface)
    text_tokenizer = CLIPTokenizer.from_pretrained(args.clip_huggingface)
    
    model.to(device)