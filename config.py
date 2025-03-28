import argparse
from dataclasses import dataclass, field
from typing import Optional
import torch 
import torchvision 
from torchvision import transforms
import os 

# ---------------- PathsConfig ----------------
@dataclass
class DataConfig:
    # My Laptop
    data_dir: str = "/Users/olivergrainge/Documents/github/Datasets"
    checkpoints_dir: str = "/Users/olivergrainge/Documents/github/TernaryTransformers/examples/checkpoints"

    # My Desktop
    #data_dir: str = "/home/oliver/Documents/github/TernaryTransformers/
    #examples/data"
    #checkpoints_dir: str = (
    #    "/home/oliver/Documents/github/TernaryTransformers/examples/checkpoints"
    #)

    #tokenizer_name: str = "bert-base-uncased"
    #batch_size: int = 32
   # num_workers: int = 0
    
    #context_length: int = 196
    #mlm_probability: float = 0.15

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
    max_epochs: int = 100
    accelerator: str = "auto"
    log_every_n_steps: int = 10
    val_check_interval: int = 1.0
    learning_rate: float = 0.001
    

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




# ---------------------------------------- MNIST DEFAULT CONFIG ----------------------------------------

@dataclass
class MLPModelConfig(ModelConfig):
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
class MNISTTrainConfig(TrainConfig):
    project_name: str = "MNIST-classification"
    max_epochs: int = 10
    accelerator: str = "auto"
    log_every_n_steps: int = 10
    val_check_interval: int = 0.2
    learning_rate: float = 0.001
    precision: str = "bf16"


@dataclass
class MNISTDataConfig(DataConfig):
    checkpoints_dir: str = os.path.join(DataConfig.checkpoints_dir, "mnist")
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False

    transform: torchvision.transforms.Compose = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        lambda x: x.view(-1, 784).squeeze(),  # MNIST images are 28x28 = 784 pixels
    ])



# ---------------------------------------- CIFAR10 DEFAULT CONFIG ----------------------------------------

class MiniViTModelConfig(ModelConfig):
    backbone_type = "ViT"
    transformer_heads = 4
    transformer_dim = 128
    transformer_ffn_dim = 384
    transformer_depth = 6
    transformer_dropout = 0.1
    transformer_dim_head = transformer_dim // transformer_heads
    image_size = 32
    image_channels = 3
    image_patch_size = 4
    embedding_dropout = 0.0

    embedding_norm_layer = "LayerNorm"
    embedding_linear_layer = "Linear"
    attention_linear_layer = "Linear"
    attention_norm_layer = "LayerNorm"
    feedforward_linear_layer = "Linear"
    feedforward_norm_layer = "LayerNorm"
    attention_activation_layer = "GELU"
    feedforward_activation_layer = "GELU"

    head_type: str = "ImageClassificationHead"
    head_in_dim = 128
    head_out_dim = 10
    head_dim = 128
    head_linear_layer = "Linear"
    head_depth = 1
    head_dropout = 0.0


class CIFAR10TrainConfig(TrainConfig):
    project_name = "CIFAR10-classification"
    max_epochs = 100
    learning_rate = 0.001
    log_every_n_steps = 10 
    val_check_interval = 0.2
    precision: str = "bf16"
    accelerator = "auto"


class CIFAR10DataConfig(DataConfig):
    checkpoints_dir: str = os.path.join(DataConfig.checkpoints_dir, "cifar10")
    pin_memory: bool = False
    batch_size = 128
    num_workers = 0

    transform: torchvision.transforms.Compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )



# ---------------------------------------- BertMLM DEFAULT CONFIG ----------------------------------------

class MiniBertModelConfig(ModelConfig):
    backbone_type = "Bert"
    head_type = "MLMHead"
    vocab_size = 30522
    max_seq_len = 128
    num_segments = 2
    transformer_dim = 256
    transformer_depth = 6
    transformer_heads = 8
    transformer_ffn_dim = 1024
    transformer_dim_head = transformer_dim // transformer_heads
    transformer_dropout = 0.1

    attention_norm_layer = "LayerNorm"
    feedforward_norm_layer = "LayerNorm"
    attention_activation_layer = "GELU"
    feedforward_activation_layer = "GELU"
    attention_linear_layer = "Linear"
    feedforward_linear_layer = "Linear"
    embedding_dropout = 0.1
    embedding_norm_layer = "LayerNorm"


    head_type: str = "ProjectionHead"
    head_in_dim = transformer_dim
    head_out_dim = transformer_dim
    head_dim = transformer_dim
    head_linear_layer = "Linear"
    head_depth = 1
    head_dropout = 0.0




class BertMLMTrainConfig(TrainConfig):
    project_name = "WikiText2-MLM"
    learning_rate = 1e-4
    tokenizer_name = "bert-base-uncased"
    precision = "bf16"
    val_check_interval = 0.2
    max_epochs = 10 
    accelerator = "auto"
    log_every_n_steps = 10


class WikiTextMLMDataConfig(DataConfig):
    data_dir = os.path.join(DataConfig.data_dir, "wikitext")
    mlm_probability = 0.15
    num_workers = 0
    batch_size = 12
    context_length = 128 
    tokenizer_name = "bert-base-uncased"
    pin_memory = False



# ---------------------------------------- AutoLM DEFAULT CONFIG ----------------------------------------


class NanoGPTModelConfig(ModelConfig):
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
    attention_activation_layer = "GELU"
    feedforward_linear_layer = "Linear"
    feedforward_norm_layer = "LayerNorm"
    feedforward_activation_layer = "GELU"

    head_type = "ProjectionHead"
    head_dim = transformer_dim
    head_linear_layer = "Linear"
    head_in_dim = transformer_dim
    head_out_dim = vocab_size


class ShakespeareTrainConfig(TrainConfig):
    project_name = "Shakespeare-AutoLM"
    max_epochs = 50
    learning_rate = 3e-4
    accelerator = "auto"
    precision = "bf16"
    log_every_n_steps = 10
    val_check_interval = 0.2
    gradient_clip_val = 1.0
    accumulate_grad_batches = 1


class ShakespeareDataConfig(DataConfig):
    checkpoints_dir: str = os.path.join(DataConfig.checkpoints_dir, "tiny_shakespeare")
    tokenizer_name = "gpt2"
    batch_size = 12 
    num_workers = 0 
    context_length = 196 
    pin_memory = False 

if __name__ == "__main__":
    model_config, training_config, data_config = parse_configs(
        ModelConfig, TrainConfig, DataConfig
    )
    print(model_config)
    print(training_config)
    print(data_config)
