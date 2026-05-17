# FGSM for WiSwin (Widar3.0 BVP)

This mini-pack adds an FGSM attacker that plugs into your existing pipeline:

- **Dataloader** yields `(bvps, labels, bvp_len)` where `bvps` is `(B, T, 20, 20)`.
- **Model** forward is `logits = model(bvps, bvp_len)`.

## Files

- `attacks/fgsm.py`: `FGSMConfig`, `FGSMAttack`, and a helper for evaluation.
- `scripts/fgsm_eval.py`: evaluate clean vs FGSM accuracy.
- `scripts/fgsm_adv_train.py`: FGSM adversarial training (optional).

## Quick start

1) Put `attacks/` and `scripts/` into your project root.

2) Ensure your best model exists, e.g. `checkpoints/model.pth`.

3) Evaluate robustness:

```bash
python scripts/fgsm_eval.py --weights checkpoints/fgsm_adv_best.pth --eps 0.02 --setting 1 --num_workers 0
```

4) (Optional) adversarial training:

```bash
python scripts/fgsm_adv_train.py --eps 0.02 --adv_ratio 0.5 --epochs 30 --setting 1 --num_workers 0 \
  --init_weights checkpoints/model.pth
```

## Choosing eps

Because your BVP is min-max normalized to ~[0,1], typical eps values are:

- 0.005, 0.01 (very small)
- 0.02, 0.05 (common)
- 0.1 (strong, may be unrealistic)

You can sweep eps and report the accuracy drop.
```bash
python scripts/fgsm_sweep.py --weights checkpoints/pgd_adv_best.pth --setting 1 --num_workers 0
```