import argparse
import os
import sys

import torch
from datasets import load_dataset

import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from trainers import WikiText2BertMLMTrainer
from config import ModelConfig, TrainConfig, DataConfig, parse_configs


class WikiText2BertMLMModelConfig(ModelConfig):
    backbone_type = "Bert"
    head_type = "MLMHead"
    vocab_size = 30522
    max_seq_len = 128
    num_segments = 2
    transformer_dim = 256
    transformer_depth = 6
    transformer_heads = 8
    transformer_mlp_dim = 1024
    transformer_dim_head = transformer_dim // transformer_heads
    transformer_dropout = 0.1
    transformer_emb_dropout = 0.1
    transformer_num_segments = 2
    transformer_attention_norm_layer = "LayerNorm"
    transformer_feedforward_norm_layer = "LayerNorm"
    transformer_attention_activation_layer = "GELU"
    transformer_feedforward_activation_layer = "GELU"
    transformer_attention_linear_layer = "Linear"
    transformer_feedforward_linear_layer = "Linear"

class WikiText2BertMLMTrainConfig(TrainConfig):
    project_name = "WikiText2-MLM"
    batch_size = 48
    learning_rate = 1e-4
    total_train_samples = 100_000
    total_val_samples = 5_000
    tokenizer_name = "bert-base-uncased"
    mlm_probability = 0.15
    precision = "bf16"
    num_workers = 4

class WikiText2BertMLMDataConfig(DataConfig):
    data_dir = os.path.join(DataConfig.data_dir, "wikitext")



def main():
    model_config, train_config, data_config = parse_configs(
        WikiText2BertMLMModelConfig,
        WikiText2BertMLMTrainConfig,
        WikiText2BertMLMDataConfig,
    )

    # Load the dataset first
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    module = WikiText2BertMLMTrainer(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
    )

    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        accelerator=train_config.accelerator,
        precision=train_config.precision,
        logger=pl.loggers.WandbLogger(
            project=train_config.project_name, name=module.experiment_name
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints/bert/",
                filename=f"{module.experiment_name}-{{epoch}}-{{val_loss:.2f}}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
        ],
    )

    trainer.fit(module)


if __name__ == "__main__":
    main()
