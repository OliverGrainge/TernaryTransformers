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
    backbone_type = "CausalTransformer"
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


class TinyShakespeareTrainConfig(TrainConfig):
    project_name = "TinyShakespeare"
    batch_size = 64
    max_epochs = 50
    learning_rate = 3e-4
    block_size = 64

    start_text = "O Romeo, Romeo, "
    max_tokens = 100
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
    )

    trainer.fit(module)


    # Test the model after training
    model = module.eval()
    sample_text = model.generate(
        start_text=train_config.start_text,
        max_tokens=train_config.max_tokens,
        temperature=train_config.temperature,
    )
    print("\nGenerated text:")
    print(sample_text)


if __name__ == "__main__":
    main()
