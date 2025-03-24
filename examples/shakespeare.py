import torch
import os
import sys
import wandb
from pytorch_lightning import Trainer
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers import TinyShakespeareTrainer
import pytorch_lightning as pl


def parse_args():
    parser = argparse.ArgumentParser(description='Shakespeare Text Generation Training Script')
    
    # Model configuration
    parser.add_argument('--backbone', type=str, default='CausalTransformer', help='Backbone architecture')
    parser.add_argument('--head', type=str, default='projection', help='Head architecture')
    parser.add_argument('--vocab-size', type=int, default=65, help='Vocabulary size (will be adjusted by dataset)')
    parser.add_argument('--max-seq-len', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--dim', type=int, default=384, help='Model dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--mlp-dim', type=int, default=1536, help='MLP dimension')
    parser.add_argument('--dim-head', type=int, default=64, help='Dimension per head')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--emb-dropout', type=float, default=0.1, help='Embedding dropout rate')
    
    # Layer types
    parser.add_argument('--attention-norm', type=str, default='LayerNorm', help='Attention normalization layer')
    parser.add_argument('--feedforward-norm', type=str, default='LayerNorm', help='Feedforward normalization layer')
    parser.add_argument('--attention-activation', type=str, default='GELU', help='Attention activation layer')
    parser.add_argument('--feedforward-activation', type=str, default='GELU', help='Feedforward activation layer')
    parser.add_argument('--attention-linear', type=str, default='Linear', help='Attention linear layer')
    parser.add_argument('--feedforward-linear', type=str, default='Linear', help='Feedforward linear layer')
    
    # Training configuration
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--block-size', type=int, default=64, help='Block size for training')
    parser.add_argument('--max-epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--accelerator', type=str, default=None, help='Accelerator (cpu, gpu, etc.)')
    parser.add_argument('--project-name', type=str, default='TinyShakespeare', help='W&B project name')
    parser.add_argument('--run-name', type=str, default='shakespeare_transformer', help='W&B run name')
    
    # Generation configuration
    parser.add_argument('--start-text', type=str, default='O Romeo, Romeo, ', help='Starting text for generation')
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    module = TinyShakespeareTrainer(
        backbone=args.backbone,
        head=args.head,
        backbone_kwargs={
            "vocab_size": args.vocab_size,  # This will be automatically adjusted by the dataset
            "max_seq_len": args.max_seq_len,
            "dim": args.dim,
            "depth": args.depth,
            "heads": args.heads,
            "mlp_dim": args.mlp_dim,
            "dim_head": args.dim_head,
            "dropout": args.dropout,
            "emb_dropout": args.emb_dropout,
            "attention_norm_layer": args.attention_norm,
            "feedforward_norm_layer": args.feedforward_norm,
            "attention_activation_layer": args.attention_activation,
            "feedforward_activation_layer": args.feedforward_activation,
            "attention_linear_layer": args.attention_linear,
            "feedforward_linear_layer": args.feedforward_linear,
        },
        head_kwargs={"in_dim": args.dim, "out_dim": args.vocab_size},  # Will be automatically adjusted
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_epochs=args.max_epochs,
    )

    trainer = pl.Trainer(
        max_epochs=module.hparams.max_epochs,
        accelerator=args.accelerator,
        logger=pl.loggers.WandbLogger(
            project=args.project_name, name=args.run_name
        ),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints/shakespeare/",
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
        start_text=args.start_text, max_tokens=args.max_tokens, temperature=args.temperature
    )
    print("\nGenerated text:")
    print(sample_text)


if __name__ == "__main__":
    main()
