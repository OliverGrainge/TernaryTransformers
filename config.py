import argparse
from dataclasses import dataclass, field
from typing import Optional

# ---------------- PathsConfig ----------------
@dataclass
class DataConfig:
    data_dir: str = "/Users/olivergrainge/Documents/github/Datasets"
    checkpoints_dir: str = "/Users/olivergrainge/Documents/github/TernaryTransformers/examples/checkpoints"

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--data_dir", type=str, default=None,
                            help="Path to datasets directory")
        parser.add_argument("--checkpoints_dir", type=str, default=None,
                            help="Path to checkpoints directory")

    @classmethod
    def from_args(cls, args, defaults: "DataConfig"):
        return cls(
            data_dir=args.data_dir if args.data_dir is not None else defaults.data_dir,
            checkpoints_dir=args.checkpoints_dir if args.checkpoints_dir is not None else defaults.checkpoints_dir,
        )
    
    @classmethod
    def from_parser(cls):
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        args = parser.parse_args()
        return cls.from_args(args)

# ---------------- TrainingConfig ----------------
@dataclass
class TrainingConfig:
    project_name: str
    batch_size: int = 64
    max_epochs: int = 100
    accelerator: str = "gpu"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('training')
        group.add_argument('--project-name', type=str, default="default_project")
        group.add_argument('--batch-size', type=int, default=cls.batch_size)
        group.add_argument('--max-epochs', type=int, default=cls.max_epochs)
        group.add_argument('--accelerator', type=str, default=cls.accelerator)

    @classmethod
    def from_args(cls, args):
        return cls(
            project_name=args.project_name,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            accelerator=args.accelerator
        )
    
    @classmethod
    def from_parser(cls):
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        args = parser.parse_args()
        return cls.from_args(args)

# ---------------- BackboneConfig ----------------
@dataclass
class BackboneConfig:
    backbone: str = "minivit"  # backbone architecture name
    depth: int = 12     # number of layers
    heads: int = 8      # number of attention heads 
    ffn_dim: int = 2048   # dimension of the feedforward network
    dim: int = 256       # dimension of the model
    image_size: int = 224  # size of the image in (H, W)
    patch_size: int = 16   # patch size in images
    in_channels: int = 3  # number of channels in the image
    dim_head: int = 64  # dimension of each attention head
    dropout: float = 0.1  # dropout rate
    emb_dropout: float = 0.1  # dropout rate for embeddings
    embedding_norm: str = "LayerNorm"  # normalization layer for embeddings
    embedding_linear: str = "Linear"  # linear layer for embeddings
    attention_linear_layer: str = "Linear"  # linear layer for attention
    attention_norm_layer: str = "LayerNorm"  # normalization layer for attention
    feedforward_linear_layer: str = "Linear"  # linear layer for feedforward
    feedforward_norm_layer: str = "LayerNorm"  # normalization layer for feedforward
    attention_activation_layer: str = "GELU"  # activation layer for attention
    feedforward_activation_layer: str = "GELU"  # activation layer for feedforward
    in_dim: int = 256  # dimension of the input
    vocab_size: int = 1000  # vocabulary size
    max_seq_len: int = 196  # maximum sequence length

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--backbone", type=str, default=cls.backbone,
                          help="Backbone architecture name")
        parser.add_argument("--depth", type=int, default=cls.depth,
                          help="Number of layers")
        parser.add_argument("--heads", type=int, default=cls.heads,
                          help="Number of attention heads")
        parser.add_argument("--ffn_dim", type=int, default=cls.ffn_dim,
                          help="Feedforward network dimension")
        parser.add_argument("--dim", type=int, default=cls.dim,
                          help="Model dimension")
        parser.add_argument("--image_size", type=int, default=cls.image_size,
                          help="Image size (assumes square)")
        parser.add_argument("--patch_size", type=int, default=cls.patch_size,
                          help="Patch size in images")
        parser.add_argument("--in_channels", type=int, default=cls.in_channels,
                          help="Number of channels in image")
        parser.add_argument("--dim_head", type=int, default=cls.dim_head,
                          help="Dimension of each attention head")
        parser.add_argument("--dropout", type=float, default=cls.dropout,
                          help="Dropout rate")
        parser.add_argument("--emb_dropout", type=float, default=cls.emb_dropout,
                          help="Embedding dropout rate")
        parser.add_argument("--embedding_norm", type=str, default=cls.embedding_norm,
                          help="Normalization layer for embeddings")
        parser.add_argument("--embedding_linear", type=str, default=cls.embedding_linear,
                          help="Linear layer for embeddings")
        parser.add_argument("--attention_linear_layer", type=str, default=cls.attention_linear_layer,
                          help="Linear layer for attention")
        parser.add_argument("--attention_norm_layer", type=str, default=cls.attention_norm_layer,
                          help="Normalization layer for attention")
        parser.add_argument("--feedforward_linear_layer", type=str, default=cls.feedforward_linear_layer,
                          help="Linear layer for feedforward")
        parser.add_argument("--feedforward_norm_layer", type=str, default=cls.feedforward_norm_layer,
                          help="Normalization layer for feedforward")
        parser.add_argument("--attention_activation_layer", type=str, default=cls.attention_activation_layer,
                          help="Activation layer for attention")
        parser.add_argument("--feedforward_activation_layer", type=str, default=cls.feedforward_activation_layer,
                          help="Activation layer for feedforward")
        parser.add_argument("--in_dim", type=int, default=cls.in_dim,
                          help="Dimension of the input")
        parser.add_argument("--vocab_size", type=int, default=cls.vocab_size,
                          help="Vocabulary size")
        parser.add_argument("--max_seq_len", type=int, default=cls.max_seq_len,
                          help="Maximum sequence length")

    @classmethod
    def from_args(cls, args, defaults: "BackboneConfig"):
        return cls(
            backbone=args.backbone if args.backbone is not None else defaults.backbone,
            depth=args.depth if args.depth is not None else defaults.depth,
            heads=args.heads if args.heads is not None else defaults.heads,
            ffn_dim=args.ffn_dim if args.ffn_dim is not None else defaults.ffn_dim,
            dim=args.dim if args.dim is not None else defaults.dim,
            image_size=args.image_size if args.image_size is not None else defaults.image_size,
            patch_size=args.patch_size if args.patch_size is not None else defaults.patch_size,
            in_channels=args.in_channels if args.in_channels is not None else defaults.in_channels,
            dim_head=args.dim_head if args.dim_head is not None else defaults.dim_head,
            dropout=args.dropout if args.dropout is not None else defaults.dropout,
            emb_dropout=args.emb_dropout if args.emb_dropout is not None else defaults.emb_dropout,
            embedding_norm=args.embedding_norm if args.embedding_norm is not None else defaults.embedding_norm,
            embedding_linear=args.embedding_linear if args.embedding_linear is not None else defaults.embedding_linear,
            attention_linear_layer=args.attention_linear_layer if args.attention_linear_layer is not None else defaults.attention_linear_layer,
            attention_norm_layer=args.attention_norm_layer if args.attention_norm_layer is not None else defaults.attention_norm_layer,
            feedforward_linear_layer=args.feedforward_linear_layer if args.feedforward_linear_layer is not None else defaults.feedforward_linear_layer,
            feedforward_norm_layer=args.feedforward_norm_layer if args.feedforward_norm_layer is not None else defaults.feedforward_norm_layer,
            attention_activation_layer=args.attention_activation_layer if args.attention_activation_layer is not None else defaults.attention_activation_layer,
            feedforward_activation_layer=args.feedforward_activation_layer if args.feedforward_activation_layer is not None else defaults.feedforward_activation_layer,
            in_dim=args.in_dim if args.in_dim is not None else defaults.in_dim,
            vocab_size=args.vocab_size if args.vocab_size is not None else defaults.vocab_size,
            max_seq_len=args.max_seq_len if args.max_seq_len is not None else defaults.max_seq_len,
        )

# ---------------- HeadConfig ----------------
@dataclass
class HeadConfig:
    head: str = "ImageClassificationHead"
    in_dim: int = 256
    out_dim: int = 1000
    linear_layer: str = "Linear"
    dim: int = 256
    depths: int = 2
    dropout: float = 0.1

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument("--head", type=str, default=cls.head,
                          help="Head architecture name")
        parser.add_argument("--head_in_dim", type=int, default=cls.in_dim,
                          help="Head input dimension")
        parser.add_argument("--head_out_dim", type=int, default=cls.out_dim,
                          help="Head output dimension")
        parser.add_argument("--head_linear_layer", type=str, default=cls.linear_layer,
                          help="Head linear layer type")
        parser.add_argument("--head_dim", type=int, default=cls.dim,
                          help="Head dimension")
        parser.add_argument("--head_depths", type=int, default=cls.depths,
                          help="Head number of layers")
        parser.add_argument("--head_dropout", type=float, default=cls.dropout,
                          help="Head dropout rate")

# ---------------- ModelConfig ----------------
@dataclass
class ModelConfig:
    # Backbone parameters
    backbone: str = "vit"
    depth: int = 12
    heads: int = 8
    ffn_dim: int = 2048
    dim: int = 256
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    dim_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.1
    embedding_norm: str = "LayerNorm"
    embedding_linear: str = "Linear"
    attention_linear_layer: str = "Linear"
    attention_norm_layer: str = "LayerNorm"
    feedforward_linear_layer: str = "Linear"
    feedforward_norm_layer: str = "LayerNorm"
    attention_activation_layer: str = "GELU"
    feedforward_activation_layer: str = "GELU"
    in_dim: int = 256
    vocab_size: int = 1000
    max_seq_len: int = 196

    # Head parameters
    head: str = "ImageClassificationHead"
    head_in_dim: int = 256
    head_out_dim: int = 1000
    head_linear_layer: str = "Linear"
    head_dim: int = 256
    head_depths: int = 2
    head_dropout: float = 0.1

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        # Backbone arguments
        parser.add_argument("--backbone", type=str, default=cls.backbone,
                          help="Backbone architecture name")
        parser.add_argument("--depth", type=int, default=cls.depth,
                          help="Number of layers")
        parser.add_argument("--heads", type=int, default=cls.heads,
                          help="Number of attention heads")
        parser.add_argument("--ffn_dim", type=int, default=cls.ffn_dim,
                          help="Feedforward network dimension")
        parser.add_argument("--dim", type=int, default=cls.dim,
                          help="Model dimension")
        parser.add_argument("--image_size", type=int, default=cls.image_size,
                          help="Image size (assumes square)")
        parser.add_argument("--patch_size", type=int, default=cls.patch_size,
                          help="Patch size in images")
        parser.add_argument("--in_channels", type=int, default=cls.in_channels,
                          help="Number of channels in image")
        parser.add_argument("--dim_head", type=int, default=cls.dim_head,
                          help="Dimension of each attention head")
        parser.add_argument("--dropout", type=float, default=cls.dropout,
                          help="Dropout rate")
        parser.add_argument("--emb_dropout", type=float, default=cls.emb_dropout,
                          help="Embedding dropout rate")
        parser.add_argument("--embedding_norm", type=str, default=cls.embedding_norm,
                          help="Normalization layer for embeddings")
        parser.add_argument("--embedding_linear", type=str, default=cls.embedding_linear,
                          help="Linear layer for embeddings")
        parser.add_argument("--attention_linear_layer", type=str, default=cls.attention_linear_layer,
                          help="Linear layer for attention")
        parser.add_argument("--attention_norm_layer", type=str, default=cls.attention_norm_layer,
                          help="Normalization layer for attention")
        parser.add_argument("--feedforward_linear_layer", type=str, default=cls.feedforward_linear_layer,
                          help="Linear layer for feedforward")
        parser.add_argument("--feedforward_norm_layer", type=str, default=cls.feedforward_norm_layer,
                          help="Normalization layer for feedforward")
        parser.add_argument("--attention_activation_layer", type=str, default=cls.attention_activation_layer,
                          help="Activation layer for attention")
        parser.add_argument("--feedforward_activation_layer", type=str, default=cls.feedforward_activation_layer,
                          help="Activation layer for feedforward")
        parser.add_argument("--in_dim", type=int, default=cls.in_dim,
                          help="Dimension of the input")
        parser.add_argument("--vocab_size", type=int, default=cls.vocab_size,
                          help="Vocabulary size")
        parser.add_argument("--max_seq_len", type=int, default=cls.max_seq_len,
                          help="Maximum sequence length")

        # Head arguments
        parser.add_argument("--head", type=str, default=cls.head,
                          help="Head architecture name")
        parser.add_argument("--head_in_dim", type=int, default=cls.head_in_dim,
                          help="Head input dimension")
        parser.add_argument("--head_out_dim", type=int, default=cls.head_out_dim,
                          help="Head output dimension")
        parser.add_argument("--head_linear_layer", type=str, default=cls.head_linear_layer,
                          help="Head linear layer type")
        parser.add_argument("--head_dim", type=int, default=cls.head_dim,
                          help="Head dimension")
        parser.add_argument("--head_depths", type=int, default=cls.head_depths,
                          help="Head number of layers")
        parser.add_argument("--head_dropout", type=float, default=cls.head_dropout,
                          help="Head dropout rate")

    @classmethod
    def from_args(cls, args):
        return cls(
            # Backbone parameters
            backbone=args.backbone,
            depth=args.depth,
            heads=args.heads,
            ffn_dim=args.ffn_dim,
            dim=args.dim,
            image_size=args.image_size,
            patch_size=args.patch_size,
            in_channels=args.in_channels,
            dim_head=args.dim_head,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout,
            embedding_norm=args.embedding_norm,
            embedding_linear=args.embedding_linear,
            attention_linear_layer=args.attention_linear_layer,
            attention_norm_layer=args.attention_norm_layer,
            feedforward_linear_layer=args.feedforward_linear_layer,
            feedforward_norm_layer=args.feedforward_norm_layer,
            attention_activation_layer=args.attention_activation_layer,
            feedforward_activation_layer=args.feedforward_activation_layer,
            in_dim=args.in_dim,
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            
            # Head parameters
            head=args.head,
            head_in_dim=args.head_in_dim,
            head_out_dim=args.head_out_dim,
            head_linear_layer=args.head_linear_layer,
            head_dim=args.head_dim,
            head_depths=args.head_depths,
            head_dropout=args.head_dropout
        )
    
    @classmethod
    def from_parser(cls):
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        args = parser.parse_args()
        return cls.from_args(args)

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        # Add arguments from all configs
        data_group = parser.add_argument_group('data')
        DataConfig.add_args(data_group)
        
        training_group = parser.add_argument_group('training')
        TrainingConfig.add_args(training_group)
        
        model_group = parser.add_argument_group('model')
        ModelConfig.add_args(model_group)

    @classmethod
    def from_args(cls, args):
        return cls(
            data=DataConfig.from_args(args, DataConfig()),
            training=TrainingConfig.from_args(args),
            model=ModelConfig.from_args(args)
        )

    @classmethod
    def from_parser(cls):
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        args = parser.parse_args()
        return cls.from_args(args)

# Usage:
if __name__ == "__main__":
    config = Config.from_parser()
    
    # Access all configurations
    print(f"Data directory: {config.data.data_dir}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Model depth: {config.model.depth}")