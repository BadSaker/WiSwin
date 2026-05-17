# README_ADV_TRAIN.md

按当前最终结构：

- attacks/fgsm.py
- attacks/pgd.py
- scripts/fgsm_adv_train.py
- scripts/pgd_adv_train.py
- scripts/_adv_train_common.py
- README_ADV_TRAIN.md

## 1. 设计原则
这套结构只保留 **两个攻击文件**：
- `attacks/fgsm.py`
- `attacks/pgd.py`

它们同时服务于：
1. 对抗攻击评测（eval / sweep）
2. 防御训练 / 对抗训练（adv_train）

也就是说：
- **不再区分“对抗攻击版 attack”与“对抗训练版 attack”**
- 同一个 `generate()` 接口既能给评测脚本用，也能给对抗训练脚本用

其他脚本结构维持清晰分工：
- 评测：`fgsm_eval.py / pgd_eval.py / *_sweep.py`
- 防御训练：`fgsm_adv_train.py / pgd_adv_train.py`

## 2. 原理
对抗训练的训练目标为：

```text
loss = (1 - adv_ratio) * CE(model(x), y) + adv_ratio * CE(model(x_adv), y)
```

其中：
- `x`：干净样本
- `x_adv`：由 `attacks/fgsm.py` 或 `attacks/pgd.py` 在线生成的对抗样本

### FGSM
```text
x_adv = x + eps * sign(∂L/∂x)
```

### PGD
```text
x_adv^(t+1) = Project( x_adv^t + alpha * sign(∂L/∂x_adv^t) )
```

## 3. 为什么这版更适合当前项目
- 只保留两个 attack 文件，不会越改越乱
- 你原来的攻击评测脚本也能继续用
- 新增对抗训练脚本时，不需要再引入新的 attack 命名空间
- 统一接口，组内维护成本最低

## 4. 使用方法

### 4.1 FGSM 对抗训练
```bash
python scripts/fgsm_adv_train.py --init_weights checkpoints/model.pth --eps 0.005 --adv_ratio 0.5 --epochs 30 --setting 1 --num_workers 0
```

### 4.2 PGD 对抗训练
```bash
python scripts/pgd_adv_train.py --init_weights checkpoints/model.pth --eps 0.005 --steps 20 --adv_ratio 0.5 --epochs 30 --setting 1 --num_workers 0
```

## 5. 推荐实验流程
1. `train.py` 训练干净模型，得到 `checkpoints/model.pth`
2. `fgsm_adv_train.py` / `pgd_adv_train.py` 继续做防御训练
3. 用你原来的 `fgsm_eval.py` / `pgd_eval.py` 验证鲁棒性提升
