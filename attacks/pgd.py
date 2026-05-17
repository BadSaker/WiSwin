# attacks/pgd.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
try:
    from typing import Literal  # Py>=3.8
except Exception:  # Py<3.8
    try:
        from typing_extensions import Literal  # type: ignore
    except Exception:
        Literal = str  # type: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PGDConfig:
    # Threat model
    eps: float = 0.02                 # L∞ or L2 budget in input space
    alpha: Optional[float] = None     # step size; if None, use 2*eps/steps (common heuristic)
    steps: int = 10                   # number of PGD iterations
    norm: Literal["linf", "l2"] = "linf"

    # Init
    random_start: bool = True         # start from random point within epsilon ball

    # Input bounds (your BVP is normalized to ~[0,1])
    clip_min: float = 0.0
    clip_max: float = 1.0

    # Variable-length padding handling
    keep_padding: bool = True         # do NOT perturb padded frames (padding stays unchanged)
    ensure_sorted: bool = True        # if lengths not sorted (desc), sort for pack_padded_sequence
    force_eval: bool = True           # keep whole model eval(), but keep RNN modules train() for cuDNN backward

    # Targeted attack (optional)
    targeted: bool = False            # False: untargeted (maximize CE); True: targeted (minimize CE to target)


def _forward(model: nn.Module, x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    """Compatible forward: model(x, lengths) or model(x)."""
    if lengths is None:
        return model(x)
    try:
        return model(x, lengths)
    except TypeError:
        return model(x)


def _make_padding_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    x: [B,T,H,W]
    lengths: [B] (CPU or GPU)
    return: [B,T,1,1] float mask with 1 for valid frames else 0
    """
    B, T = x.shape[0], x.shape[1]
    t = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # [B,T]
    valid = (t < lengths.to(x.device).unsqueeze(1)).float()         # [B,T]
    return valid.unsqueeze(-1).unsqueeze(-1)                        # [B,T,1,1]


def _ensure_sorted(
    x: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Ensure lengths are sorted descending for pack_padded_sequence(enforce_sorted=True).

    Returns:
        x_s, y_s, lengths_s, perm_dev, inv_dev
    where perm_dev / inv_dev are indices on the SAME device as x (safe for CUDA indexing).
    lengths_s stays on CPU (recommended for many PyTorch RNN utilities).
    """
    # lengths is typically CPU; keep it that way
    if torch.all(lengths[:-1] >= lengths[1:]):
        return x, y, lengths, None, None

    perm_cpu = torch.argsort(lengths, descending=True)  # CPU
    perm_dev = perm_cpu.to(x.device)
    # inverse permutation on device
    inv_dev = torch.empty_like(perm_dev)
    inv_dev[perm_dev] = torch.arange(perm_dev.numel(), device=perm_dev.device)

    x_s = x.index_select(0, perm_dev)
    y_s = y.index_select(0, perm_dev)
    lengths_s = lengths.index_select(0, perm_cpu)  # keep CPU
    return x_s, y_s, lengths_s, perm_dev, inv_dev


def _project_linf(x0: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    delta = torch.clamp(x - x0, min=-eps, max=eps)
    return x0 + delta


def _project_l2(x0: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    # per-sample projection onto L2 ball
    delta = x - x0
    B = delta.shape[0]
    flat = delta.view(B, -1)
    norms = torch.norm(flat, p=2, dim=1, keepdim=True).clamp(min=1e-12)
    factors = torch.clamp(eps / norms, max=1.0)
    delta = (flat * factors).view_as(delta)
    return x0 + delta


def _rand_init(x0: torch.Tensor, cfg: PGDConfig) -> torch.Tensor:
    if cfg.norm == "linf":
        r = torch.empty_like(x0).uniform_(-cfg.eps, cfg.eps)
        return x0 + r
    else:  # l2
        B = x0.shape[0]
        r = torch.randn_like(x0)
        flat = r.view(B, -1)
        norms = torch.norm(flat, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        # random radius in [0, eps]
        u = torch.rand(B, 1, device=x0.device)
        flat = flat / norms * (u * cfg.eps)
        return x0 + flat.view_as(x0)


def perturb_stats(
    x0: torch.Tensor,
    x_adv: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    keep_padding: bool = True,
) -> Dict[str, Any]:
    """Return simple perturbation statistics for sanity-checking."""
    delta = x_adv - x0
    if keep_padding and lengths is not None:
        mask = _make_padding_mask(x0, lengths)
        delta = delta * mask

    B = delta.shape[0]
    flat = delta.view(B, -1)
    linf = flat.abs().max(dim=1).values  # [B]
    l2 = torch.norm(flat, p=2, dim=1)   # [B]
    return {
        "linf_mean": linf.mean().item(),
        "linf_max": linf.max().item(),
        "l2_mean": l2.mean().item(),
        "l2_max": l2.max().item(),
    }


class PGDAttack:
    """
    WiSwin-compatible PGD attacker.

    Input:
        x: [B,T,20,20]
        lengths: [B] (CPU recommended)

    Output:
        x_adv: same shape as x, clipped into [clip_min, clip_max]
    """

    def __init__(self, cfg: PGDConfig):
        if cfg.steps < 1:
            raise ValueError("steps must be >= 1")
        self.cfg = cfg

    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        y_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.cfg

        if cfg.targeted and y_target is None:
            raise ValueError("targeted=True but y_target is None")

        x0 = x.detach()
        perm_dev = None
        inv_dev = None
        if lengths is not None and cfg.ensure_sorted:
            x0, y, lengths, perm_dev, inv_dev = _ensure_sorted(x0, y, lengths)
            if y_target is not None and perm_dev is not None:
                y_target = y_target.index_select(0, perm_dev)

        # alpha default: 2*eps/steps (common)
        alpha = cfg.alpha if cfg.alpha is not None else (2.0 * cfg.eps / float(cfg.steps))

        # init
        if cfg.random_start:
            x_adv = _rand_init(x0, cfg)
        else:
            x_adv = x0.clone()

        # padding mask
        mask = None
        if cfg.keep_padding and lengths is not None:
            mask = _make_padding_mask(x0, lengths)
            x_adv = x0 + (x_adv - x0) * mask  # keep padding unchanged

        # clip to input range
        x_adv = x_adv.clamp(cfg.clip_min, cfg.clip_max).detach()

        # --- cuDNN RNN backward requires RNN modules in training mode ---
        was_training = model.training
        rnn_states = []
        try:
            if cfg.force_eval:
                model.eval()
                for m in model.modules():
                    if isinstance(m, nn.RNNBase):
                        rnn_states.append((m, m.training))
                        m.train(True)

            for _ in range(cfg.steps):
                x_adv = x_adv.detach().requires_grad_(True)

                logits = _forward(model, x_adv, lengths)
                if cfg.targeted:
                    loss = F.cross_entropy(logits, y_target)
                    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                    grad = -grad  # targeted -> gradient descent
                else:
                    loss = F.cross_entropy(logits, y)
                    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                    # untargeted -> ascent (use +grad)

                if mask is not None:
                    grad = grad * mask

                if cfg.norm == "linf":
                    x_adv = x_adv + alpha * grad.sign()
                    x_adv = _project_linf(x0, x_adv, cfg.eps)
                else:  # l2
                    B = x_adv.shape[0]
                    g = grad.view(B, -1)
                    g_norm = torch.norm(g, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                    g_unit = (g / g_norm).view_as(x_adv)
                    x_adv = x_adv + alpha * g_unit
                    x_adv = _project_l2(x0, x_adv, cfg.eps)

                if mask is not None:
                    x_adv = x0 + (x_adv - x0) * mask

                x_adv = x_adv.clamp(cfg.clip_min, cfg.clip_max)

            if inv_dev is not None:
                x_adv = x_adv.index_select(0, inv_dev)

            return x_adv.detach()

        finally:
            if cfg.force_eval:
                for m, st in rnn_states:
                    m.train(st)
                model.train(was_training)


@torch.no_grad()
def pgd_eval_batch(
    model: nn.Module,
    attacker: PGDAttack,
    bvps: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
):
    """Return clean_correct, adv_correct, batch_size."""
    logits = model(bvps, lengths)
    pred = logits.argmax(1)
    clean_correct = (pred == labels).sum().item()

    with torch.enable_grad():
        bvps_adv = attacker.generate(model, bvps, labels, lengths)

    logits_adv = model(bvps_adv, lengths)
    pred_adv = logits_adv.argmax(1)
    adv_correct = (pred_adv == labels).sum().item()

    return clean_correct, adv_correct, labels.numel()
