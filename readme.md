# <div align="center"> Outfit-Transformer </div>

<div align="center"> 2023. 12. 26 : Train is Available </div>

## 🤗 Introduction
Implementation of paper - [Outfit Transformer: Outfit Representations for Fashion Recommendation](https://arxiv.org/abs/2204.04812)

## 🎯 Performance
### CP(Compatibility Prediction)
|Model|AUC|
|:-|:-:|
|Type-Aware|0.86|
|SCE-Net|0.91|
|CSA-Net|0.91|
|OutfitTransformer(Paper)|0.93|
|**OutfitTransformer(Implemented)**|**0.91**|

### FITB(Fill in The Blank)
|Model|AUC|
|:-|:-:|
|Type-Aware|57.83|
|SCE-Net|59.07|
|CSA-Net|63.73|
|OutfitTransformer(Paper)|67.10|
|**OutfitTransformer(Implemented)**|**?**|

## 🧱 Train
### Data Preparation
```
```

### Pretraining on CP(Compatibiliby Prediction) task
```
python train.py --task compatibility --train_batch 64 --valid_batch 96 --n_epochs 5 --learning_rate 1e-3 --work_dir $WORK_DIR --data_dir $DATA_DIR --wandb_api_key $WANDB_API_KEY
```

### Finetuning on FITB(Fill in The Blank) task
```
python train.py --task fitb --train_batch 64 --valid_batch 96 --n_epochs 3 --learning_rate 1e-5 --work_dir $WORK_DIR --data_dir $DATA_DIR --wandb_api_key $WANDB_API_KEY
```

## 🔍 Test
```
python test.py
```

## 🧶 Inference
```
python inference.py
```

## 🔔 Note
- A paper review of each implementation can be found at [here](). (Only Available in Korean)
- This is **NON-OFFICIAL** implementation. (The official repo has not been released.)
