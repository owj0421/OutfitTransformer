# <div align="center"> Outfit-Transformer </div>

## 🤗 Introduction
Implementation of paper - [Outfit Transformer: Outfit Representations for Fashion Recommendation](https://arxiv.org/abs/2204.04812)<br>

⚠️ The original paper outlines the specifics of the target item for Compatitible Item Retrieval (CIR) and Fill-in-the-Blank (FITB). Nonetheless, for the sake of impartial evaluation alongside other models, this information was intentionally excluded. (Should a dataset emerge that necessitates the prediction of a matching item when presented with a description unrelated to the target item itself, the model will be retrained accordingly.)

<div align="center"> <img src = https://github.com/owj0421/outfit-transformer/assets/98876272/fc39d1c7-b076-495d-8213-3b98ef038b64 width = 512> </div>

## 🎯 Performance
The figures below are derived using the Polyvore test dataset.

<div align="center">

|Model|CP(AUC)|FITB(Accuracy)|CIR(Recall@10)|
|:-|-:|-:|-:|
|Type-Aware|0.86|57.83|3.50|
|SCE-Net|0.91|59.07|5.10|
|CSA-Net|0.91|63.73|8.27|
|OutfitTransformer(Paper)|0.93|67.10|9.58|
|**Implemented <br> (w/o target desc.)**|**0.91**|**64.10**|Not Trained|
</div>



## ⚙ Install Dependencies
This code is tested with python 3.9.16, torch 1.12.1
```
python -m pip install -r requirements.txt
```

## 🧱 Train
### Data Preparation
Download the polyvore dataset from [here](https://github.com/xthan/polyvore-dataset)

### Pretraining on CP(Compatibiliby Prediction) task
```
python train.py --task cp --train_batch 64 --valid_batch 96 --n_epochs 5 --learning_rate 1e-5 --scheduler_step_size 1000 --work_dir $WORK_DIR --data_dir $DATA_DIR --wandb_api_key $WANDB_API_KEY
```

### Finetuning on CIR(Complementary Item Retrival) task
```
python train.py --task cir --train_batch 64 --valid_batch 96 --n_epochs 5 --learning_rate 1e-5 --scheduler_step_size 1000 --work_dir $WORK_DIR --data_dir $DATA_DIR --wandb_api_key $WANDB_API_KEY --checkpoint $CHECKPOINT
```

## 🔍 Test
```
python test.py --task $TASK --polyvore_split nondisjoint --test_batch 96 --data_dir $DATA_DIR --checkpoint $CHECKPOINT
```

## 🧶 Checkpoints
Download the checkpoint from [here](https://drive.google.com/drive/folders/1etD-c-BOgvDHTHkmQoNh3K31P_MWt2eg?usp=share_link)

## 🔔 Note
- A paper review of implementation can be found at [here](). (Only Available in Korean)
- This is **NON-OFFICIAL** implementation. (The official repo has not been released.)
