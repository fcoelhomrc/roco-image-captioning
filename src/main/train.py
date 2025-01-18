import argparse

import yaml
import pprint
import time
import os

import torch
import torch.utils.data
import lightning as L

import wandb
from pytorch_lightning.loggers import WandbLogger

from src.lib.dataset.ROCODataset import ImageTextDataset
from src.lib.data_transform.collate import collate_clip
from utils.monitoring_tools import create_directory

from src.lib.model.clip import MedCLIP


parser = argparse.ArgumentParser(
    description='CLIP fine-tuning on medical data'
)

parser.add_argument('--config',
                    required=True,
                    help='Config file (YAML)')


args = parser.parse_args()

with open(args.config, "r") as yaml_file:
    parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
print("Current configuration:")
pprint.pprint(parameters)
print("-"*80)

wandb_dir = create_directory(
    base_name=parameters["registry"]["name"],
    root_dir=parameters["registry"]["wandb_root_dir"],
)

wandb_logger = WandbLogger(
    project=parameters["registry"]["project"],
    save_dir=os.path.join(str(parameters["registry"]["wandb_root_dir"]), os.path.basename(wandb_dir)),
    config=parameters,
)

print("Loading train data...")
start = time.perf_counter()
train_data = ImageTextDataset(
    "train", cfg=parameters,
)

print(f"Successfully loaded train data! Elapsed: {time.perf_counter() - start:.1f} seconds")

print("Loading validation data...")
start = time.perf_counter()
val_data = ImageTextDataset(
    "val", cfg=parameters,
)
print(f"Successfully loaded validation data! Elapsed: {time.perf_counter() - start:.1f} seconds")

print("Loading test data...")
start = time.perf_counter()
test_data = ImageTextDataset(
    "test", cfg=parameters,
)
print(f"Successfully loaded test data! Elapsed: {time.perf_counter() - start:.1f} seconds")


train_dataloader = torch.utils.data.DataLoader(
    batch_size=parameters['train']['batch_size'],
    dataset=train_data,
    shuffle=True,
    num_workers=4,
    drop_last=True,
    collate_fn=collate_clip,
)

val_dataloader = torch.utils.data.DataLoader(
    batch_size=parameters['train']['batch_size'],
    dataset=val_data,
    num_workers=4,
    drop_last=True,
    collate_fn=collate_clip,
)

test_dataloader = torch.utils.data.DataLoader(
    batch_size=parameters['train']['batch_size'],
    dataset=test_data,
    num_workers=4,
    drop_last=True,
    collate_fn=collate_clip,
)


# Prepare callbacks
callback_model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=parameters['registry']['checkpoints_dir'],
    filename='{epoch}',
    monitor='val/loss',
    save_last=True,
    save_top_k=1,
)

callback_early_stopping = L.pytorch.callbacks.EarlyStopping(
    monitor='val/loss',
    patience=parameters['train']['early_stopping_patience'],
)

callback_learning_rate_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')

callbacks = [
    callback_model_checkpoint,
    callback_early_stopping,
    callback_learning_rate_monitor,
]

# Prepare model for training
model = MedCLIP(user_parameters=parameters)
print("Created new instance of model...")

torch.set_float32_matmul_precision('medium')
print("Warning! Set float32 matmul precision to 'medium'...")


trainer = L.Trainer(
    limit_train_batches=parameters['train']['limit_train_batches'],
    max_epochs=parameters['train']['max_epochs'],
    logger=wandb_logger,
    callbacks=callbacks,
    log_every_n_steps=parameters['train']['log_every_n_steps'],
    gradient_clip_val=parameters['train']['gradient_clip_val'],
)

trainer.fit(model, train_dataloader, val_dataloader)
trainer.test(dataloaders=test_dataloader, ckpt_path="best")


wandb.finish()
