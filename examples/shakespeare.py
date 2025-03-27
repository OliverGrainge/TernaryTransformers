import argparse
import os
import sys

import torch
from pytorch_lightning import Trainer

import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from trainers import TinyShakespeareTrainer
from config import ModelConfig, TrainConfig, DataConfig, parse_configs


class TinyShakespeareModelConfig(ModelConfig):
    backbone_type = "gpt"
    vocab_size = 65 
    max_seq_len = 64
    transformer_dim = 384
    transformer_depth = 6
    transformer_heads = 6
    transformer_dim_head = transformer_dim // transformer_heads
    transformer_ffn_dim = 1536
    transformer_dropout = 0.1
    embedding_dropout = 0.1

    embedding_norm_layer = "LayerNorm"
    embedding_linear_layer = "Linear"
    attention_linear_layer = "Linear"
    attention_norm_layer = "LayerNorm"
    feedforward_linear_layer = "Linear"
    feedforward_norm_layer = "LayerNorm"
    feedforward_activation_layer = "GELU"

    head_type = "ProjectionHead"
    head_dim = transformer_dim
    head_linear_layer = "Linear"
    head_in_dim = transformer_dim 
    head_out_dim = vocab_size


class TinyShakespeareTrainConfig(TrainConfig):
    project_name = "TinyShakespeare"
    batch_size = 64
    max_epochs = 50
    learning_rate = 3e-4
    block_size = 64

    start_text = "O Romeo, Romeo, "
    max_tokens = 100
    val_check_interval = 1.0
    temperature = 0.8

class TinyShakespeareDataConfig(DataConfig):
    checkpoints_dir: str = os.path.join(DataConfig.checkpoints_dir, "tiny_shakespeare")



def main():
    model_config, train_config, data_config = parse_configs(TinyShakespeareModelConfig, TinyShakespeareTrainConfig, TinyShakespeareDataConfig)

    module = TinyShakespeareTrainer(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
    )

    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        accelerator=train_config.accelerator,
        logger=pl.loggers.WandbLogger(project=train_config.project_name, name=module.experiment_name),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=data_config.checkpoints_dir,
                filename="shakespeare-{epoch}-{val_loss:.2f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
        ],
        val_check_interval=train_config.val_check_interval,
    )

    trainer.fit(module)



if __name__ == "__main__":
    main()
