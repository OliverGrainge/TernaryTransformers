import argparse
from dataclasses import dataclass, field
from typing import Optional


# ---------------- PathsConfig ----------------
@dataclass
class DataConfig:
    # My Laptop
    # data_dir: str = "/Users/olivergrainge/Documents/github/Datasets"
    # checkpoints_dir: str = "/Users/olivergrainge/Documents/github/TernaryTransformers/examples/checkpoints"

    # My Desktop
    data_dir: str = "/home/oliver/Documents/github/TernaryTransformers/examples/data"
    checkpoints_dir: str = (
        "/home/oliver/Documents/github/TernaryTransformers/examples/checkpoints"
    )

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


# ---------------- TrainConfig ----------------
@dataclass
class TrainConfig:
    project_name: str = "empty"
    batch_size: int = 64
    max_epochs: int = 100
    accelerator: str = "auto"
    log_steps: int = 10
    val_check_interval: int = 1.0
    learning_rate: float = 0.001
    num_workers: int = 0

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


# ---------------- ModelConfig ----------------
@dataclass
class ModelConfig:
    # Base architecture parameters
    backbone: str = "vit"  # backbone architecture name

    # Transformer architecture parameters
    transformer_depth: int = 12
    transformer_heads: int = 8
    transformer_dim: int = 256  # renamed from dim
    transformer_dim_head: int = transformer_dim // transformer_heads
    transformer_ffn_dim: int = 2048  # renamed from ffn_dim
    transformer_dropout: float = 0.1  # renamed from dropout

    # MLP architecture parameters
    mlp_in_dim: int = 256
    mlp_depth: int = 3
    mlp_dim: int = 256
    mlp_ffn_dim: int = 2048
    mlp_dropout: float = 0.1

    # Image-specific parameters
    image_size: int = 224
    image_patch_size: int = 16
    image_channels: int = 3

    # Embedding parameters
    embedding_dim: int = 256  # renamed from in_dim
    embedding_dropout: float = 0.1  # renamed from emb_dropout
    max_sequence_length: int = 196  # renamed from max_seq_len
    vocab_size: int = 1000

    # Attention parameters
    attention_head_dim: int = 64  # renamed from attn_head_dim

    # Layer specifications
    embedding_norm_layer: str = "LayerNorm"  # renamed for consistency
    embedding_linear_layer: str = "Linear"  # renamed for consistency
    attention_linear_layer: str = "Linear"
    attention_norm_layer: str = "LayerNorm"
    attention_activation_layer: str = "GELU"
    feedforward_linear_layer: str = "Linear"
    feedforward_norm_layer: str = "LayerNorm"
    feedforward_activation_layer: str = "GELU"
    mlp_linear_layer: str = "Linear"
    mlp_norm_layer: str = "LayerNorm"
    mlp_activation_layer: str = "RELU"

    # Classification head parameters
    head_type: str = "ImageClassificationHead"  # renamed from head
    head_in_dim: int = 256
    head_out_dim: int = 1000  # renamed from out_dim
    head_dim: int = 256  # renamed from head_dim
    head_linear_layer: str = "Linear"  # renamed from linear_layer
    head_depth: int = 1
    head_dropout: float = 0.0

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


def parse_configs(ModelConfig, TrainConfig, DataConfig):
    args = argparse.ArgumentParser()
    ModelConfig.add_args(args)
    TrainConfig.add_args(args)
    DataConfig.add_args(args)
    args = args.parse_args()
    return (
        ModelConfig.from_args(args),
        TrainConfig.from_args(args),
        DataConfig.from_args(args),
    )


if __name__ == "__main__":
    model_config, training_config, data_config = parse_configs(
        ModelConfig, TrainConfig, DataConfig
    )
    print(model_config)
    print(training_config)
    print(data_config)
