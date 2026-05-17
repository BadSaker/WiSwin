# README_PGD_v3.md

这是 WiSwin（Widar3.0 BVP）工程的 **PGD 攻击封装 v2（已检查与完善）**。

## 1. 复制文件
将压缩包内容复制到你的项目根目录：

- attacks/pgd.py
- scripts/pgd_eval.py
- scripts/pgd_sweep.py

> 注意：本 v2 脚本已内置 PROJECT_ROOT + sys.path 注入，
> **即使在 PyCharm/VSCode 里工作目录不正确，也能正常 import 与找到默认权重路径。**

## 2. 单点 PGD 评测
```bash
python scripts/pgd_eval.py --eps 0.005 --steps 10 --setting 1 --num_workers 0
```

默认权重会自动使用：
- <项目根>/checkpoints/model.pth

你也可以显式指定：
```bash
python scripts/pgd_eval.py --weights checkpoints/model.pth --eps 0.005 --steps 10
```

可选：打印扰动统计（用于 sanity-check）
```bash
python scripts/pgd_eval.py --eps 0.005 --steps 10 --report_perturb
```

## 3. eps–acc 曲线（扫 eps）
```bash
python scripts/pgd_sweep.py --steps 10 --setting 1 --num_workers 0
```
```bash
python scripts/pgd_sweep.py --weights checkpoints/pgd_adv_best.pth --steps 10 --setting 1 --num_workers 0
```
输出：
- checkpoints/pgd_sweep.csv
- checkpoints/pgd_eps_acc.png

自定义 eps：
```bash
python scripts/pgd_sweep.py --eps_list "0.0005,0.001,0.002,0.005,0.01,0.02"
```

输出 csv 时也记录扰动统计：
```bash
python scripts/pgd_sweep.py --steps 10 --report_perturb
```

## 4. 关键实现细节（避免 cuDNN RNN backward 报错）
你的 WiSwin 使用 `nn.LSTM`，而 cuDNN 的 RNN backward 只能在 training mode 调用。
本实现采取：
- `model.eval()`（关掉 dropout/drop-path，梯度更稳定）
- 但 **仅将 RNN/LSTM 子模块临时 train()**（确保能对输入求梯度）
- 攻击结束后恢复原状态

## 5. 参数建议（与你的 FGSM 曲线对齐）
由于 PGD 更强，建议从你 FGSM 的脆弱阈值附近开始：
- eps: 0.001 / 0.002 / 0.005
- steps: 10（基线）/ 20（更强）
- random_start: True（默认更强）


## 6. Python 版本兼容
如果你使用的是 Python<3.8（会出现 `cannot import name 'Literal' from typing`），请安装：

```bash
pip install typing_extensions
```

或升级 Python 至 3.8+。
