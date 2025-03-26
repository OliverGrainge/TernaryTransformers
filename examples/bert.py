import torch
import os
import sys
import wandb
from datasets import load_dataset
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers import WikiText2BertMLMTrainer
import pytorch_lightning as pl


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


def main():
    args = parse_args()

    # Load the dataset first
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    except Exception as e:
        print("Error loading dataset. Trying with force_download=True...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", force_download=True)

    module = WikiText2BertMLMTrainer(
        backbone=args.backbone,
        head=args.head,
        backbone_kwargs={
            "vocab_size": args.vocab_size,
            "max_seq_len": args.max_seq_len,
            "dim": args.dim,
            "depth": args.depth,
            "heads": args.heads,
            "mlp_dim": args.mlp_dim,
            "dim_head": args.dim_head,
            "dropout": args.dropout,
            "emb_dropout": args.emb_dropout,
            "num_segments": args.num_segments,
            "attention_norm_layer": args.attention_norm,
            "feedforward_norm_layer": args.feedforward_norm,
            "attention_activation_layer": args.attention_activation,
            "feedforward_activation_layer": args.feedforward_activation,
            "attention_linear_layer": args.attention_linear,
            "feedforward_linear_layer": args.feedforward_linear,
        },
        head_kwargs={"dim": args.dim, "vocab_size": args.vocab_size},
        batch_size=args.batch_size,
        dataset=dataset,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=pl.loggers.WandbLogger(
            project=args.project_name, name=module.experiment_name
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
