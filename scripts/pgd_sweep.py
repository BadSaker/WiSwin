# scripts/pgd_sweep.py
import argparse
import sys
from pathlib import Path
import csv

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

    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--norm", type=str, default="linf", choices=["linf", "l2"])

    rs = p.add_mutually_exclusive_group()
    rs.add_argument("--random_start", action="store_true")
    rs.add_argument("--no_random_start", action="store_true")

    p.add_argument("--eps_list", type=str, default="0.0005,0.001,0.002,0.005,0.01,0.02,0.05")
    p.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    p.add_argument("--report_perturb", action="store_true", help="also save perturb stats columns into csv")
    return p.parse_args()


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """
    将输入路径解析为最终绝对路径：
    - 若本身是绝对路径，直接返回
    - 若是相对路径，优先按项目根目录拼接
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


@torch.no_grad()
def eval_clean(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y, lengths in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x, lengths)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def eval_pgd(model, loader, device, cfg: PGDConfig):
    attacker = PGDAttack(cfg)
    clean_correct, adv_correct, total = 0, 0, 0

    # optional stats
    st_sum = {"linf_mean": 0.0, "linf_max": 0.0, "l2_mean": 0.0, "l2_max": 0.0}
    st_batches = 0

    for x, y, lengths in loader:
        x = x.to(device)
        y = y.to(device)
        cc, ac, n = pgd_eval_batch(model, attacker, x, y, lengths)
        clean_correct += cc
        adv_correct += ac
        total += n

        if getattr(cfg, "_report_perturb", False):
            with torch.enable_grad():
                x_adv = attacker.generate(model, x, y, lengths)
            st = perturb_stats(x, x_adv, lengths=lengths, keep_padding=cfg.keep_padding)
            for k in st_sum:
                st_sum[k] += st[k]
            st_batches += 1

    cc_acc = clean_correct / max(total, 1)
    ac_acc = adv_correct / max(total, 1)

    if st_batches > 0:
        for k in st_sum:
            st_sum[k] /= st_batches

    return cc_acc, ac_acc, st_sum, st_batches


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = resolve_path(args.out_dir, PROJECT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps_list = [float(s.strip()) for s in args.eps_list.split(",") if s.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("PROJECT_ROOT      :", PROJECT_ROOT)
    print("Current cwd       :", Path.cwd())
    print("args.weights(raw) :", args.weights)
    print("args.out_dir(raw) :", args.out_dir)
    print("device            :", device)
    print("=" * 60)

    _, test_loader = build_dataloader(
        batch_size=args.batch_size, num_workers=args.num_workers, setting=args.setting
    )

    net = WiSwin().to(device)

    weights_path = resolve_path(args.weights, PROJECT_ROOT)
    print("Resolved weights  :", weights_path)
    print("Weights exists    :", weights_path.exists())

    if not weights_path.exists():
        raise FileNotFoundError(
            f"权重文件不存在：{weights_path}\n"
            f"当前工作目录：{Path.cwd()}\n"
            f"项目根目录：{PROJECT_ROOT}\n"
            f"原始参数 --weights：{args.weights}"
        )

    state = torch.load(str(weights_path), map_location=device)
    net.load_state_dict(state)
    net.eval()

    random_start = True
    if args.no_random_start:
        random_start = False
    elif args.random_start:
        random_start = True

    clean_acc = eval_clean(net, test_loader, device)
    print(f"Clean Acc: {clean_acc:.4f}")

    rows = []
    for eps in eps_list:
        cfg = PGDConfig(
            eps=eps,
            alpha=args.alpha,
            steps=args.steps,
            norm=args.norm,
            random_start=random_start,
        )
        if args.report_perturb:
            cfg._report_perturb = True  # internal flag

        cc, ac, st, _ = eval_pgd(net, test_loader, device, cfg)
        drop = cc - ac
        alpha_eff = cfg.alpha if cfg.alpha is not None else (2.0 * cfg.eps / float(cfg.steps))

        print(f"eps={eps:.6f}  adv_acc={ac:.4f}  drop={drop:.4f}")
        rows.append((eps, cc, ac, drop, args.steps, alpha_eff, args.norm, random_start, st))

    csv_path = out_dir / "pgd_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["eps", "clean_acc", "adv_acc", "acc_drop", "steps", "alpha", "norm", "random_start"]
        if args.report_perturb:
            header += ["linf_mean", "linf_max", "l2_mean", "l2_max"]
        w.writerow(header)

        for eps, cc, ac, drop, steps, alpha_eff, norm, rs, st in rows:
            row = [eps, cc, ac, drop, steps, alpha_eff, norm, rs]
            if args.report_perturb:
                row += [st["linf_mean"], st["linf_max"], st["l2_mean"], st["l2_max"]]
            w.writerow(row)

    print(f"Saved: {csv_path}")

    try:
        import matplotlib.pyplot as plt

        xs = [r[0] for r in rows]
        ys = [r[2] for r in rows]

        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xscale("log")
        plt.xlabel("epsilon (log scale)")
        plt.ylabel("PGD adv accuracy")
        plt.title(f"PGD eps-acc (steps={args.steps}, norm={args.norm}, random_start={random_start})")

        fig_path = out_dir / "pgd_eps_acc.png"
        plt.savefig(fig_path, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()