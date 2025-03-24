import torch
import os
import sys
import wandb
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers import MNISTTrainer
import pytorch_lightning as pl


def parse_args():
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    
    # Model configuration
    parser.add_argument('--backbone', type=str, default='mlp', help='Backbone architecture')
    parser.add_argument('--head', type=str, default='none', help='Head architecture')
    parser.add_argument('--in-dim', type=int, default=784, help='Input dimension')
    parser.add_argument('--mlp-dim', type=int, default=512, help='MLP hidden dimension')
    parser.add_argument('--out-dim', type=int, default=10, help='Output dimension')
    parser.add_argument('--linear-layer', type=str, default='tlinear_group', help='Linear layer type')
    parser.add_argument('--activation', type=str, default='RELU', help='Activation function')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--norm-layer', type=str, default='layernorm', help='Normalization layer')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--max-epochs', type=int, default=10, help='Maximum number of epochs')
    parser.add_argument('--accelerator', type=str, default='cpu', help='Accelerator (cpu, gpu, etc.)')
    parser.add_argument('--project-name', type=str, default='mnist-classification', help='W&B project name')
    parser.add_argument('--log-steps', type=int, default=5, help='Log every N steps')
    parser.add_argument('--val-check-interval', type=float, default=0.25, help='Validation check interval')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    module = MNISTTrainer(
        backbone=args.backbone,
        head=args.head,
        backbone_kwargs={
            "in_dim": args.in_dim,
            "mlp_dim": args.mlp_dim,
            "out_dim": args.out_dim,
            "linear_layer": args.linear_layer,
            "activation_layer": args.activation,
            "num_layers": args.num_layers,
            "norm_layer": args.norm_layer,
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
                dirpath="checkpoints/mnist/",
                filename=f"{module.experiment_name}-{{epoch}}-{{val_loss:.2f}}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
        ],
        log_every_n_steps=args.log_steps,
        val_check_interval=args.val_check_interval,
    )

    trainer.fit(module)


if __name__ == "__main__":
    main()
