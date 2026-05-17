"""Evaluate WiSwin robustness under FGSM.

Run from project root:
    python scripts/fgsm_eval.py --weights checkpoints/model.pth --eps 0.02

It reports clean accuracy and FGSM accuracy on your validation set.
"""

from __future__ import annotations

import argparse

import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_loader
import model as wiswin_model
from attacks.fgsm import FGSMConfig, FGSMAttack, fgsm_eval_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="checkpoints/model.pth")
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--setting", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = data_loader.build_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        setting=args.setting,
    )

    net = wiswin_model.WiSwin().to(device)
    state = torch.load(args.weights, map_location=device)
    net.load_state_dict(state)
    net.eval()

    attacker = FGSMAttack(FGSMConfig(eps=args.eps))

    clean_correct = 0
    adv_correct = 0
    total = 0

    for bvps, labels, lengths in test_loader:
        bvps = bvps.to(device)
        labels = labels.to(device)
        cc, ac, n = fgsm_eval_batch(net, attacker, bvps, labels, lengths)
        clean_correct += cc
        adv_correct += ac
        total += n

    clean_acc = clean_correct / total
    adv_acc = adv_correct / total
    drop = clean_acc - adv_acc
    print(f"Clean Acc: {clean_acc:.4f}")
    print(f"FGSM  Acc (eps={args.eps}): {adv_acc:.4f}")
    print(f"Accuracy drop: {drop:.4f}")


if __name__ == "__main__":
    main()
