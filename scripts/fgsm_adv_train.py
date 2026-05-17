"""Adversarial training (FGSM) for WiSwin.

This script trains WiSwin with a mixture of clean and FGSM-perturbed samples.

Example:
    python scripts/fgsm_adv_train.py --eps 0.02 --adv_ratio 0.5 --epochs 30

Notes:
- Default clamp range is [0,1] which matches the min-max normalization in your loader.
- FGSM is crafted on-the-fly with current model weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import data_loader
import model as wiswin_model
from attacks.fgsm import FGSMConfig, FGSMAttack


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup_epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--adv_ratio", type=float, default=0.5, help="portion of loss from adversarial samples")
    p.add_argument("--init_weights", type=str, default="", help="optional clean-trained weights")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, attacker, adv_ratio, device):
    model.train()
    total_loss = 0.0
    total_correct = 0

    for bvps, labels, lengths in loader:
        bvps = bvps.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Clean
        logits_clean = model(bvps, lengths)
        loss_clean = criterion(logits_clean, labels)

        # Adversarial (need gradients for bvps)
        with torch.enable_grad():
            x_adv = attacker.generate(model, bvps, labels, lengths)
        logits_adv = model(x_adv, lengths)
        loss_adv = criterion(logits_adv, labels)

        loss = (1.0 - adv_ratio) * loss_clean + adv_ratio * loss_adv
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * bvps.size(0)
        total_correct += (logits_clean.argmax(1) == labels).sum().item()

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    for bvps, labels, lengths in loader:
        bvps = bvps.to(device)
        labels = labels.to(device)
        logits = model(bvps, lengths)
        loss = criterion(logits, labels)
        total_loss += loss.item() * bvps.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"model_fgsm_eps{args.eps:g}_adv{args.adv_ratio:g}.pth"

    train_loader, test_loader = data_loader.build_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        setting=args.setting,
    )

    net = wiswin_model.WiSwin().to(device)
    if args.init_weights:
        net.load_state_dict(torch.load(args.init_weights, map_location=device))
        print(f"Loaded init weights: {args.init_weights}")

    optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))

    def warmup_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    attacker = FGSMAttack(FGSMConfig(eps=args.eps, force_eval=True, keep_padding=True))
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    print(f"Start FGSM adversarial training: eps={args.eps}, adv_ratio={args.adv_ratio}")

    for epoch in range(args.epochs):
        lr_now = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            net, train_loader, optimizer, criterion, attacker, args.adv_ratio, device
        )
        val_loss, val_acc = validate(net, test_loader, criterion, device)

        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        print(
            f"Epoch {epoch+1}/{args.epochs} | LR: {lr_now:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), best_path)
            print(f"🎉 Best val acc: {best_val_acc:.4f}. Saved to {best_path}")


if __name__ == "__main__":
    main()
