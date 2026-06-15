import argparse
import math
import os
import time

import numpy as np
import torch

from cs336_basics.module import (
    AdamW,
    TransformerLM,
    cross_entropy,
    get_batch,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def cosine_lr_schedule(step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def evaluate(model, dataset, batch_size, context_length, device, num_batches=10):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = get_batch(dataset, batch_size, context_length, device)
            logits = model(inputs)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    # Model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    # Optimizer
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    # Device
    default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load datasets with memmap for memory efficiency (avoids loading full file into RAM)
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r") if args.val_data else None

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from checkpoint at step {start_step}")

    model.train()
    t0 = time.time()

    for step in range(start_step, args.total_steps):
        # Update learning rate with cosine schedule
        lr = cosine_lr_schedule(step, args.lr, args.min_lr, args.warmup_steps, args.total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            print(f"step {step+1:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            t0 = time.time()

        if val_data is not None and (step + 1) % args.val_interval == 0:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, args.device)
            val_ppl = math.exp(val_loss)
            print(f"  val loss {val_loss:.4f} | val ppl {val_ppl:.2f}")

        if (step + 1) % args.checkpoint_interval == 0:
            path = os.path.join(args.checkpoint_dir, f"ckpt_{step+1:07d}.pt")
            save_checkpoint(model, optimizer, step + 1, path)
            print(f"  saved checkpoint to {path}")


if __name__ == "__main__":
    main()
