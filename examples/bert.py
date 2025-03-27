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
    max_seq_len = 512
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
    batch_size = 12
    learning_rate = 1e-4
    total_train_samples = 5_000
    total_val_samples = 1_000
    tokenizer_name = "bert-base-uncased"
    mlm_probability = 0.15

class WikiText2BertMLMDataConfig(DataConfig):
    data_dir = os.path.join(DataConfig.data_dir, "wikitext")

"""
def parse_args():
    parser = argparse.ArgumentParser(description="BERT MLM Training Script")

    # Model configuration
    parser.add_argument(
        "--backbone", type=str, default="Bert", help="Backbone architecture"
    )
    parser.add_argument("--head", type=str, default="MLMHead", help="Head architecture")
    parser.add_argument("--vocab-size", type=int, default=30522, help="Vocabulary size")
    parser.add_argument(
        "--max-seq-len", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--depth", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--mlp-dim", type=int, default=1024, help="MLP dimension")
    parser.add_argument("--dim-head", type=int, default=32, help="Dimension per head")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--emb-dropout", type=float, default=0.1, help="Embedding dropout rate"
    )
    parser.add_argument(
        "--num-segments", type=int, default=2, help="Number of segments"
    )

    # Layer types
    parser.add_argument(
        "--attention-norm",
        type=str,
        default="LayerNorm",
        help="Attention normalization layer",
    )
    parser.add_argument(
        "--feedforward-norm",
        type=str,
        default="LayerNorm",
        help="Feedforward normalization layer",
    )
    parser.add_argument(
        "--attention-activation",
        type=str,
        default="GELU",
        help="Attention activation layer",
    )
    parser.add_argument(
        "--feedforward-activation",
        type=str,
        default="GELU",
        help="Feedforward activation layer",
    )
    parser.add_argument(
        "--attention-linear", type=str, default="Linear", help="Attention linear layer"
    )
    parser.add_argument(
        "--feedforward-linear",
        type=str,
        default="Linear",
        help="Feedforward linear layer",
    )

    # Training configuration
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--max-epochs", type=int, default=10, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--accelerator", type=str, default=None, help="Accelerator (cpu, gpu, etc.)"
    )
    parser.add_argument(
        "--project-name", type=str, default="WikiText2-MLM", help="W&B project name"
    )

    return parser.parse_args()
"""

def main():
    model_config, train_config, data_config = parse_configs(
        WikiText2BertMLMModelConfig,
        WikiText2BertMLMTrainConfig,
        WikiText2BertMLMDataConfig,
    )

    # Load the dataset first

    module = WikiText2BertMLMTrainer(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
    )

    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        accelerator=train_config.accelerator,
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
