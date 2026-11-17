"""训练 DoodleMind 的端上小 CNN。

设计目标（按优先级）：
1. 纯 TS 手写推理要跑得动 → 只用 conv3x3 / ReLU / maxpool / GAP / FC，BN 全部折叠进 conv
2. int8 量化后体积 < 300KB → 参数量压在 20 万以内
3. 148 类 top-1 尽量高 → 4 段 conv + GAP，配强数据增强

关键：训练用的数据增强必须在**栅格化之后的位图上**做，且幅度要贴近真人手指画图
（平移、缩放、旋转、笔画粗细），不要做弹性变形这种线下不存在的形变。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from categories import CATEGORIES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tools" / "out"
SIZE = 32
NUM_CLASSES = len(CATEGORIES)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DoodleNet(nn.Module):
    """4 段 conv，通道 24/48/96/128，GAP + 单层 FC。"""

    def __init__(self, num_classes: int = NUM_CLASSES, width: tuple[int, int, int, int] = (24, 48, 96, 128)):
        super().__init__()
        c1, c2, c3, c4 = width
        self.b1 = ConvBlock(1, c1)
        self.b2 = ConvBlock(c1, c2)
        self.b3 = ConvBlock(c2, c3)
        self.b4 = ConvBlock(c3, c4)
        self.head = nn.Linear(c4, num_classes)

    def forward(self, x):
        x = F.max_pool2d(self.b1(x), 2)   # 32 -> 16
        x = F.max_pool2d(self.b2(x), 2)   # 16 -> 8
        x = F.max_pool2d(self.b3(x), 2)   # 8  -> 4
        x = self.b4(x)                    # 4
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


# ---------------------------------------------------------------- augmentation

def augment(xb: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """xb: (N,1,32,32) float。用 grid_sample 做随机仿射，顺序：缩放→旋转→平移。"""
    n = xb.shape[0]
    ang = torch.from_numpy(rng.uniform(-13.0, 13.0, n).astype(np.float32)) * (math.pi / 180.0)
    sc = torch.from_numpy(rng.uniform(0.82, 1.14, n).astype(np.float32))
    tx = torch.from_numpy(rng.uniform(-0.09, 0.09, n).astype(np.float32))
    ty = torch.from_numpy(rng.uniform(-0.09, 0.09, n).astype(np.float32))

    cos, sin = torch.cos(ang), torch.sin(ang)
    # 输出网格 -> 输入坐标的仿射（归一化坐标）
    theta = torch.zeros(n, 2, 3, dtype=torch.float32)
    theta[:, 0, 0] = cos / sc
    theta[:, 0, 1] = -sin / sc
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin / sc
    theta[:, 1, 1] = cos / sc
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, xb.shape, align_corners=False)
    out = F.grid_sample(xb, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # 线宽/亮度抖动（gamma）
    gam = torch.from_numpy(rng.uniform(0.72, 1.45, n).astype(np.float32)).view(n, 1, 1, 1)
    out = torch.clamp(out, 0.0, 1.0) ** gam
    if rng.random() < 0.5:
        noise = torch.from_numpy(rng.normal(0, 0.022, out.shape).astype(np.float32))
        out = torch.clamp(out + noise, 0.0, 1.0)
    return out


# --------------------------------------------------------------------- training

def load_data(per_class: int, seed: int):
    d = np.load(OUT / "doodles.npz", allow_pickle=True)
    X, y, words = d["X"], d["y"], d["words"]
    print(f"载入 {X.shape[0]} 样本 / {len(words)} 类", flush=True)

    rng = np.random.default_rng(seed)
    keep_idx = []
    for cls in range(len(words)):
        idx = np.where(y == cls)[0]
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep_idx.append(idx)
    keep = np.concatenate(keep_idx)
    X = X[keep].astype(np.float32) / 255.0
    y = y[keep].astype(np.int64)

    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n_val = max(2000, int(len(y) * 0.05))
    return X[n_val:], y[n_val:], X[:n_val], y[:n_val], words


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, dev: torch.device, bs: int = 512):
    model.eval()
    top1 = top5 = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i : i + bs]).unsqueeze(1).to(dev)
            logits = model(xb)
            yb = torch.from_numpy(y[i : i + bs]).to(dev)
            k = min(5, logits.shape[1])
            _, pred = logits.topk(k, dim=1)
            top1 += (pred[:, 0] == yb).sum().item()
            top5 += (pred == yb[:, None]).any(dim=1).sum().item()
    n = len(X)
    return top1 / n, top5 / n


def resolve_device(choice: str) -> torch.device:
    """auto 走 CUDA，但要有兜底：CUDA 版 torch 装上了、驱动却认不出卡的情况常见。"""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("指定了 --device cuda，但 torch.cuda.is_available() 为 False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--width", type=str, default="24,48,96,128")
    ap.add_argument("--out", type=str, default="doodlenet.pt")
    # 默认 auto：有 CUDA 就用 CUDA。这个网很小，GPU 的瓶颈其实在数据搬运上，
    # 所以 --device cuda 时要开 pin_memory + 非阻塞拷贝，否则提升有限。
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    dev = resolve_device(args.device)
    print(f"设备 {dev}" + (f"  ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""), flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    width = tuple(int(v) for v in args.width.split(","))
    Xtr, ytr, Xva, yva, words = load_data(args.per_class, args.seed)
    print(f"训练 {Xtr.shape[0]} / 验证 {Xva.shape[0]}", flush=True)

    model = DoodleNet(len(words), width=width).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"参数量 {nparam:,}  (conv 部分 int8 ≈ {(nparam - model.head.weight.numel() - model.head.bias.numel()) / 1024:.0f} KB)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    steps_per_epoch = math.ceil(len(ytr) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.25, div_factor=10.0, final_div_factor=200.0
    )
    crit = nn.CrossEntropyLoss(label_smoothing=0.08)

    rng = np.random.default_rng(args.seed + 7)
    best = 0.0
    hist = []
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(ytr))
        t0 = time.time()
        run_loss = 0.0
        seen = 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            xb = torch.from_numpy(Xtr[idx]).unsqueeze(1)
            xb = augment(xb, rng).to(dev, non_blocking=True)
            yb = torch.from_numpy(ytr[idx]).to(dev, non_blocking=True)
            if rng.random() < 0.3:  # mixup：抑制过自信，量化后更稳
                lam = float(rng.beta(0.35, 0.35))
                pi = torch.randperm(xb.shape[0])
                xb = lam * xb + (1 - lam) * xb[pi]
                logits = model(xb)
                loss = lam * crit(logits, yb) + (1 - lam) * crit(logits, yb[pi])
            else:
                logits = model(xb)
                loss = crit(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(idx)
            seen += len(idx)
        t1 = time.time()
        t1_acc, t5_acc = evaluate(model, Xva, yva, dev)
        hist.append({"epoch": ep + 1, "loss": round(run_loss / seen, 4), "top1": round(t1_acc, 4), "top5": round(t5_acc, 4)})
        star = ""
        if t1_acc > best:
            best = t1_acc
            torch.save(
                {
                    # 权重存 CPU 副本：导出脚本在纯 CPU 上跑，
                    # 存 CUDA 张量会让它必须记得传 map_location，漏一次就炸。
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "width": list(width),
                    "words": [str(w) for w in words],
                },
                OUT / args.out,
            )
            star = "  *"
        print(
            f"ep {ep + 1:>2}/{args.epochs}  loss {run_loss / seen:.3f}  "
            f"top1 {t1_acc:.4f}  top5 {t5_acc:.4f}  {t1 - t0:.0f}s{star}",
            flush=True,
        )

    (OUT / "train_hist.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"\n最佳 top1 = {best:.4f}，权重已存 {OUT / args.out}", flush=True)


if __name__ == "__main__":
    main()
