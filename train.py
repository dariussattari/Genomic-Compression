"""
Train the genomic compressor on the human reference genome.

Training set: chr1 to chr20, chr23, chr24 (autosomes minus chr21 and chr22).
Validation:   chr21.
Held out for downstream evaluation: chr22.

The best checkpoint is selected by validation bits-per-base.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

from data import create_dataloaders
from model import DecoderOnlyTransformer

LN2 = math.log(2)
BASES_PER_TOKEN = 6


def loss_to_bpb(loss):
    return loss / LN2 / BASES_PER_TOKEN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU:    {torch.cuda.get_device_name(0)}')
        print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    return device


def build_lr_scheduler(optimizer, warmup_steps, total_steps, min_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = model(x).logits

        loss = loss_fn(logits.float().view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss, loss_to_bpb(avg_loss)


def train(args):
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader = create_dataloaders(
        args.token_dir,
        seq_len=args.seq_len,
        stride=args.seq_len,
        batch_size=args.batch_size,
    )
    print(f'Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}')

    model = DecoderOnlyTransformer(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        bias=True,
        tie_weights=True,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {args.num_layers}L / {args.embed_dim}d / {args.num_heads}H ({num_params:,} params)')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = build_lr_scheduler(optimizer, args.warmup_steps, total_steps)
    scaler = GradScaler('cuda', enabled=device.type == 'cuda')
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_bpb = float('inf')
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_tokens = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(x).logits
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * y.numel()
            running_tokens += y.numel()
            global_step += 1

            if batch_idx % args.log_every == 0:
                avg_loss = running_loss / running_tokens
                lr = scheduler.get_last_lr()[0]
                print(
                    f'  epoch {epoch + 1}/{args.epochs} '
                    f'| batch {batch_idx:>5} / {len(train_loader)} '
                    f'| loss {avg_loss:.4f} | bpb {loss_to_bpb(avg_loss):.4f} '
                    f'| lr {lr:.2e}'
                )

        train_loss = running_loss / running_tokens
        val_loss, val_bpb = evaluate(model, val_loader, device)
        elapsed = (time.time() - epoch_start) / 60

        print(
            f'Epoch {epoch + 1}/{args.epochs} done in {elapsed:.1f} min '
            f'| train bpb {loss_to_bpb(train_loss):.4f} '
            f'| val bpb {val_bpb:.4f}'
        )

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_bpb': val_bpb,
            'config': {
                'vocab_size': args.vocab_size,
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'num_layers': args.num_layers,
                'seq_len': args.seq_len,
            },
        }

        last_path = os.path.join(args.checkpoint_dir, 'last_model.pt')
        torch.save(checkpoint, last_path)

        if val_bpb < best_bpb:
            best_bpb = val_bpb
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f'  New best: val bpb {val_bpb:.4f} saved to {best_path}')

    print(f'\nTraining complete. Best val bpb: {best_bpb:.4f}')


def parse_args():
    parser = argparse.ArgumentParser(description='Train the genomic compressor')

    parser.add_argument('--token-dir', type=str, default='tokenized_data',
                        help='Directory containing chr1.npy, chr2.npy, ...')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    parser.add_argument('--vocab-size', type=int, default=4096)
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--seq-len', type=int, default=2048)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
