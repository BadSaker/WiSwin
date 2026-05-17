# attacks/fgsm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FGSMConfig:
    eps: float = 0.02
    clip_min: float = 0.0
    clip_max: float = 1.0

    keep_padding: bool = True     # 不扰动 padding 帧
    force_eval: bool = True       # 生成对抗样本时：整体 eval()（更稳定）
    ensure_sorted: bool = True    # lengths 若未降序则排序（适配 enforce_sorted=True）


def _forward(model: nn.Module, x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    """兼容 model(x, lengths) / model(x) 两种写法。"""
    if lengths is None:
        return model(x)
    try:
        return model(x, lengths)
    except TypeError:
        return model(x)


def _make_padding_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    x: [B,T,H,W]
    lengths: [B]  valid length
    return: [B,T,1,1] with 1 for valid frames
    """
    B, T = x.shape[0], x.shape[1]
    t = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # [B,T]
    valid = (t < lengths.to(x.device).unsqueeze(1)).float()         # [B,T]
    return valid.unsqueeze(-1).unsqueeze(-1)                        # [B,T,1,1]


def _ensure_sorted(
    x: torch.Tensor, y: torch.Tensor, lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """若 lengths 未降序则排序，并返回 inv_perm 用于恢复原顺序。"""
    if torch.all(lengths[:-1] >= lengths[1:]):
        return x, y, lengths, None
    perm = torch.argsort(lengths, descending=True)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return x[perm], y[perm], lengths[perm], inv


class FGSMAttack:
    """
    WiSwin 适配版 FGSM：
    - 输入 x: [B,T,20,20]
    - forward: model(x, lengths)
    - loss: CE
    """

    def __init__(self, cfg: FGSMConfig):
        self.cfg = cfg

    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.cfg

        x0 = x.detach()
        inv_perm = None
        if lengths is not None and cfg.ensure_sorted:
            x0, y, lengths, inv_perm = _ensure_sorted(x0, y, lengths)

        # 需要对输入求梯度
        x_adv = x0.clone().detach().requires_grad_(True)

        # --- 关键：cuDNN RNN backward 只能在 RNN 模块 training mode 下 ---
        was_training = model.training
        rnn_states = []

        try:
            if cfg.force_eval:
                model.eval()  # 关掉 dropout / droppath 更稳定
                # 但 LSTM/RNN 必须 train 才能 backward
                for m in model.modules():
                    if isinstance(m, nn.RNNBase):
                        rnn_states.append((m, m.training))
                        m.train(True)

            logits = _forward(model, x_adv, lengths)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

            delta = cfg.eps * grad.sign()

            if cfg.keep_padding and lengths is not None:
                delta = delta * _make_padding_mask(x0, lengths)

            x_adv2 = (x0 + delta).clamp(cfg.clip_min, cfg.clip_max).detach()

            if inv_perm is not None:
                x_adv2 = x_adv2[inv_perm]

            return x_adv2

        finally:
            if cfg.force_eval:
                for m, st in rnn_states:
                    m.train(st)
                model.train(was_training)


@torch.no_grad()
def fgsm_eval_batch(model: nn.Module, attacker: FGSMAttack,
                    bvps: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor):
    """
    返回 clean_correct_count, adv_correct_count, batch_size
    """
    logits = model(bvps, lengths)
    pred = logits.argmax(1)
    clean_correct = (pred == labels).sum().item()

    # 生成对抗样本需要梯度，所以这里手动开 grad
    with torch.enable_grad():
        bvps_adv = attacker.generate(model, bvps, labels, lengths)

    logits_adv = model(bvps_adv, lengths)
    pred_adv = logits_adv.argmax(1)
    adv_correct = (pred_adv == labels).sum().item()

    return clean_correct, adv_correct, labels.numel()