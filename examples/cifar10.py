import torch
import os
import sys
import wandb
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers import CIFAR10Trainer
import pytorch_lightning as pl


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR10 Training Script')
    
    # Model configuration
    parser.add_argument('--backbone', type=str, default='minivit', help='Backbone architecture')
    parser.add_argument('--head', type=str, default='ImageClassificationHead', help='Head architecture')
    parser.add_argument('--depth', type=int, default=2, help='Depth of the ViT')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--mlp-dim', type=int, default=384, help='MLP dimension (128*3)')
    parser.add_argument('--dim', type=int, default=128, help='Model dimension')
    parser.add_argument('--image-size', type=int, default=32, help='Input image size')
    parser.add_argument('--patch-size', type=int, default=4, help='Patch size')
    parser.add_argument('--in-channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--dim-head', type=int, default=64, help='Dimension per head')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--emb-dropout', type=float, default=0, help='Embedding dropout rate')
    
    # layer type arguments
    parser.add_argument('--embedding-norm', type=str, default='LayerNorm', help='Embedding normalization layer')
    parser.add_argument('--embedding-linear', type=str, default='Linear', help='Embedding linear layer')
    parser.add_argument('--attention-linear-layer', type=str, default='tlinear_channel', help='Attention linear layer type')
    parser.add_argument('--attention-norm-layer', type=str, default='LayerNorm', help='Attention normalization layer')
    parser.add_argument('--feedforward-linear-layer', type=str, default='tlinear_channel', help='Feedforward linear layer type')
    parser.add_argument('--feedforward-norm-layer', type=str, default='LayerNorm', help='Feedforward normalization layer')
    parser.add_argument('--attention-activation-layer', type=str, default='GELU', help='Attention activation layer')
    parser.add_argument('--feedforward-activation-layer', type=str, default='GELU', help='Feedforward activation layer')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--accelerator', type=str, default="auto", help='Accelerator (cpu, gpu, etc.)')
    parser.add_argument('--project-name', type=str, default='cifar10-classification', help='W&B project name')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    module = CIFAR10Trainer(
        backbone=args.backbone,
        head=args.head,
        backbone_kwargs={
            "depth": args.depth,
            "heads": args.heads,
            "mlp_dim": args.mlp_dim,
            "dim": args.dim,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "in_channels": args.in_channels,
            "dim_head": args.dim_head,
            "dropout": args.dropout,
            "emb_dropout": args.emb_dropout,
            "embedding_norm": args.embedding_norm,
            "embedding_linear": args.embedding_linear,
            "attention_linear_layer": args.attention_linear_layer,
            "attention_norm_layer": args.attention_norm_layer,
            "feedforward_linear_layer": args.feedforward_linear_layer,
            "feedforward_norm_layer": args.feedforward_norm_layer,
            "attention_activation_layer": args.attention_activation_layer,
            "feedforward_activation_layer": args.feedforward_activation_layer,
        },
        batch_size=args.batch_size,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=pl.loggers.WandbLogger(
            project=args.project_name, name=module.experiment_name
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints/cifar10/",
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
