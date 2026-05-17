import argparse
from pathlib import Path
import csv
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import WiSwin
from data_loader import build_dataloader
from attacks.fgsm import FGSMConfig, FGSMAttack, fgsm_eval_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=str(PROJECT_ROOT / "checkpoints" / "model.pth"))
    p.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    p.add_argument("--setting", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eps_list", type=str, default="0.0005,0.001,0.002,0.005,0.01,0.02,0.05")
    return p.parse_args()


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


def eval_fgsm(model, loader, device, eps):
    attacker = FGSMAttack(FGSMConfig(eps=eps))
    model.eval()
    clean_correct, adv_correct, total = 0, 0, 0
    for x, y, lengths in loader:
        x = x.to(device)
        y = y.to(device)
        cc, ac, n = fgsm_eval_batch(model, attacker, x, y, lengths)
        clean_correct += cc
        adv_correct += ac
        total += n
    return clean_correct / max(total, 1), adv_correct / max(total, 1)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps_list = [float(s.strip()) for s in args.eps_list.split(",") if s.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    train_loader, test_loader = build_dataloader(
        batch_size=args.batch_size, num_workers=args.num_workers, setting=args.setting
    )

    # model
    net = WiSwin().to(device)
    state = torch.load(args.weights, map_location=device)
    net.load_state_dict(state)
    net.eval()

    clean_acc = eval_clean(net, test_loader, device)
    print(f"Clean Acc: {clean_acc:.4f}")

    rows = []
    for eps in eps_list:
        cc, ac = eval_fgsm(net, test_loader, device, eps)
        drop = cc - ac
        print(f"eps={eps:.6f}  adv_acc={ac:.4f}  drop={drop:.4f}")
        rows.append((eps, cc, ac, drop))

    # write csv
    csv_path = out_dir / "fgsm_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eps", "clean_acc", "adv_acc", "acc_drop"])
        w.writerows(rows)
    print(f"Saved: {csv_path}")

    # plot (optional)
    try:
        import matplotlib.pyplot as plt
        xs = [r[0] for r in rows]
        ys = [r[2] for r in rows]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xscale("log")
        plt.xlabel("epsilon (log scale)")
        plt.ylabel("FGSM adv accuracy")
        plt.title("FGSM eps-acc curve")
        fig_path = out_dir / "fgsm_eps_acc.png"
        plt.savefig(fig_path, bbox_inches="tight")
        print(f"Saved: {fig_path}")
    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()