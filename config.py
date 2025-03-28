import argparse
from dataclasses import dataclass, field
from typing import Optional
import torch 
import torchvision 
from torchvision import transforms
import os 

# ---------------- Config ----------------
@dataclass
class Config:
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        # Add all arguments without separate grouping
        for field in cls.__dataclass_fields__.values():
            parser.add_argument(
                f"--{field.name}",
                type=field.type,
                default=getattr(cls, field.name),
                help=f"{field.name} parameter",
            )
        return parser  # Return the parser object after adding arguments

    @classmethod
    def from_args(cls, args):
        return cls(
            **{
                field.name: getattr(args, field.name)
                for field in cls.__dataclass_fields__.values()
            }
        )

    @classmethod
    def from_parser(cls):
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        args = parser.parse_args()
        return cls.from_args(args)



def parse_configs(model_config, train_config, data_config):
    parser = argparse.ArgumentParser()
    parser = model_config.add_args(parser)
    parser = train_config.add_args(parser)
    parser = data_config.add_args(parser)
    args = parser.parse_args()
    model_config = model_config.from_args(args)
    train_config = train_config.from_args(args)
    data_config = data_config.from_args(args)
    return model_config, train_config, data_config




# ---------------------------------------- MNIST DEFAULT CONFIG ----------------------------------------

@dataclass
class MLPModelConfig(Config):
    backbone_type: str = "mlp"
    mlp_in_dim: int = 784
    mlp_depth: int = 3
    mlp_dim: int = 256
    mlp_dropout: float = 0.1
    mlp_linear_layer: str = "Linear"
    mlp_norm_layer: str = "LayerNorm"
    mlp_activation_layer: str = "RELU"

    head_type: str = "ImageClassificationHead"  # renamed from head
    head_in_dim: int = 256
    head_dim = 128
    head_out_dim: int = 10  # renamed from out_dim
    head_linear_layer: str = "Linear"  # renamed from linear_layer
    head_depth: int = 1
    head_dropout: float = 0.0

@dataclass
class MNISTTrainConfig(Config):
    project_name: str = "MNIST-classification"
    max_epochs: int = 10
    accelerator: str = "auto"
    log_every_n_steps: int = 10
    val_check_interval: int = 0.2
    learning_rate: float = 0.001
    precision: str = "bf16"


@dataclass
class MNISTDataConfig(Config):
    data_dir: str = "/Users/olivergrainge/Documents/github/Datasets"
    checkpoints_dir: str = "/Users/olivergrainge/Documents/github/TernaryTransformers/examples/checkpoints"
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False

    transform: str = None



# ---------------------------------------- CIFAR10 DEFAULT CONFIG ----------------------------------------

@dataclass
class MiniViTModelConfig(Config):
    backbone_type: str = "ViT"
    transformer_heads: int = 4
    transformer_dim: int = 128
    transformer_ffn_dim: int = 384
    transformer_depth: int = 6
    transformer_dropout: float = 0.1
    transformer_dim_head: int = field(default=128 // 4)  # transformer_dim // transformer_heads
    image_size: int = 32
    image_channels: int = 3
    image_patch_size: int = 4
    embedding_dropout: float = 0.0

    embedding_norm_layer: str = "LayerNorm"
    embedding_linear_layer: str = "Linear"
    attention_linear_layer: str = "Linear"
    attention_norm_layer: str = "LayerNorm"
    feedforward_linear_layer: str = "Linear"
    feedforward_norm_layer: str = "LayerNorm"
    attention_activation_layer: str = "GELU"
    feedforward_activation_layer: str = "GELU"

    head_type: str = "ImageClassificationHead"
    head_in_dim: int = field(default=128)  # transformer_dim
    head_out_dim: int = 10
    head_dim: int = field(default=128)  # transformer_dim
    head_linear_layer: str = "Linear"
    head_depth: int = 1
    head_dropout: float = 0.0


@dataclass
class CIFAR10TrainConfig(Config):
    project_name: str = "CIFAR10-classification"
    max_epochs: int = 100
    learning_rate: float = 0.001
    log_every_n_steps: int = 10
    val_check_interval: float = 0.2
    precision: str = "bf16"
    accelerator: str = "auto"


@dataclass
class CIFAR10DataConfig(Config):
    data_dir: str = "/Users/olivergrainge/Documents/github/Datasets"
    checkpoints_dir: str = "/Users/olivergrainge/Documents/github/TernaryTransformers/examples/checkpoints"
    pin_memory: bool = False
    batch_size: int = 128
    num_workers: int = 0
    transform: str = None 



# ---------------------------------------- BertMLM DEFAULT CONFIG ----------------------------------------

@dataclass
class MiniBertModelConfig(Config):
    backbone_type: str = "Bert"
    head_type: str = "MLMHead"
    vocab_size: int = 30522
    max_seq_len: int = 128
    num_segments: int = 2
    transformer_dim: int = 256
    transformer_depth: int = 6
    transformer_heads: int = 8
    transformer_ffn_dim: int = 1024
    transformer_dim_head: int = field(default=256 // 8)  # transformer_dim // transformer_heads
    transformer_dropout: float = 0.1

    attention_norm_layer: str = "LayerNorm"
    feedforward_norm_layer: str = "LayerNorm"
    attention_activation_layer: str = "GELU"
    feedforward_activation_layer: str = "GELU"
    attention_linear_layer: str = "Linear"
    feedforward_linear_layer: str = "Linear"
    embedding_dropout: float = 0.1
    embedding_norm_layer: str = "LayerNorm"

    head_type: str = "ProjectionHead"
    head_in_dim: int = field(default=256)  # transformer_dim
    head_out_dim: int = field(default=256)  # transformer_dim
    head_dim: int = field(default=256)  # transformer_dim
    head_linear_layer: str = "Linear"
    head_depth: int = 1
    head_dropout: float = 0.0


@dataclass
class BertMLMTrainConfig(Config):
    project_name: str = "WikiText2-MLM"
    learning_rate: float = 1e-4
    tokenizer_name: str = "bert-base-uncased"
    precision: str = "bf16"
    val_check_interval: float = 0.2
    max_epochs: int = 10
    accelerator: str = "auto"
    log_every_n_steps: int = 10


@dataclass
class WikiTextMLMDataConfig(Config):
    data_dir: str = "/Users/olivergrainge/Documents/github/Datasets"
    checkpoints_dir: str = "/Users/olivergrainge/Documents/github/TernaryTransformers/examples/checkpoints"
    mlm_probability: float = 0.15
    num_workers: int = 0
    batch_size: int = 12
    context_length: int = 128
    tokenizer_name: str = "bert-base-uncased"
    pin_memory: bool = False



# ---------------------------------------- AutoLM DEFAULT CONFIG ----------------------------------------


@dataclass
class NanoGPTModelConfig(Config):
    backbone_type: str = "gpt"
    vocab_size: int = 65
    max_seq_len: int = 64
    transformer_dim: int = 384
    transformer_depth: int = 6
    transformer_heads: int = 6
    transformer_dim_head: int = field(default=384 // 6)  # transformer_dim // transformer_heads
    transformer_ffn_dim: int = 1536
    transformer_dropout: float = 0.1
    embedding_dropout: float = 0.1

    embedding_norm_layer: str = "LayerNorm"
    embedding_linear_layer: str = "Linear"
    attention_linear_layer: str = "Linear"
    attention_norm_layer: str = "LayerNorm"
    attention_activation_layer: str = "GELU"
    feedforward_linear_layer: str = "Linear"
    feedforward_norm_layer: str = "LayerNorm"
    feedforward_activation_layer: str = "GELU"

    head_type: str = "ProjectionHead"
    head_dim: int = field(default=384)  # transformer_dim
    head_linear_layer: str = "Linear"
    head_in_dim: int = field(default=384)  # transformer_dim
    head_out_dim: int = field(default=65)  # vocab_size


@dataclass
class ShakespeareTrainConfig(Config):
    project_name: str = "Shakespeare-AutoLM"
    max_epochs: int = 50
    learning_rate: float = 3e-4
    accelerator: str = "auto"
    precision: str = "bf16"
    log_every_n_steps: int = 10
    val_check_interval: float = 0.2
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1


@dataclass
class ShakespeareDataConfig(Config):
    data_dir: str = "/Users/olivergrainge/Documents/github/Datasets"
    checkpoints_dir: str = "/Users/olivergrainge/Documents/github/TernaryTransformers/examples/checkpoints"
    tokenizer_name: str = "gpt2"
    batch_size: int = 12
    num_workers: int = 0
    context_length: int = 196
    pin_memory: bool = False

