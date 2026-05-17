# scripts/pgd_eval.py
import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import WiSwin
from data_loader import build_dataloader
from attacks.pgd import PGDConfig, PGDAttack, pgd_eval_batch, perturb_stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=str(PROJECT_ROOT / "checkpoints" / "model.pth"))
    p.add_argument("--setting", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--norm", type=str, default="linf", choices=["linf", "l2"])

    rs = p.add_mutually_exclusive_group()
    rs.add_argument("--random_start", action="store_true", help="enable random start (default)")
    rs.add_argument("--no_random_start", action="store_true", help="disable random start")

    p.add_argument("--report_perturb", action="store_true", help="print perturbation statistics")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    _, test_loader = build_dataloader(
        batch_size=args.batch_size, num_workers=args.num_workers, setting=args.setting
    )

    # model
    net = WiSwin().to(device)
    state = torch.load(args.weights, map_location=device)
    net.load_state_dict(state)
    net.eval()

    # attacker
    random_start = True
    if args.no_random_start:
        random_start = False
    elif args.random_start:
        random_start = True

    cfg = PGDConfig(
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
        norm=args.norm,
        random_start=random_start,
    )
    attacker = PGDAttack(cfg)

    clean_correct, adv_correct, total = 0, 0, 0
    stat_accum = {"linf_mean": 0.0, "linf_max": 0.0, "l2_mean": 0.0, "l2_max": 0.0}
    stat_batches = 0

    for bvps, labels, lengths in test_loader:
        bvps = bvps.to(device)
        labels = labels.to(device)

        # batch eval
        cc, ac, n = pgd_eval_batch(net, attacker, bvps, labels, lengths)
        clean_correct += cc
        adv_correct += ac
        total += n

        if args.report_perturb:
            with torch.enable_grad():
                bvps_adv = attacker.generate(net, bvps, labels, lengths)
            st = perturb_stats(bvps, bvps_adv, lengths=lengths, keep_padding=cfg.keep_padding)
            for k in stat_accum:
                stat_accum[k] += st[k]
            stat_batches += 1

    clean_acc = clean_correct / max(total, 1)
    adv_acc = adv_correct / max(total, 1)
    alpha_eff = cfg.alpha if cfg.alpha is not None else (2.0 * cfg.eps / float(cfg.steps))

    print(f"Clean Acc: {clean_acc:.4f}")
    print(f"PGD   Acc (eps={args.eps}, steps={args.steps}, alpha={alpha_eff:.6f}, norm={args.norm}, random_start={cfg.random_start}): {adv_acc:.4f}")
    print(f"Accuracy drop: {clean_acc - adv_acc:.4f}")

    if args.report_perturb and stat_batches > 0:
        print("Perturbation stats (mean over batches):")
        for k, v in stat_accum.items():
            print(f"  {k}: {v / stat_batches:.6f}")


if __name__ == "__main__":
    main()
