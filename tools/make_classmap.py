"""生成「它把哪些词搞混」的地图 —— 这才是玩家真正想知道的。

第一版我用分类头权重做 PCA，出来是一团均匀的点云（前两个主成分只解释 7% 的方差：
分类头的 148 个方向本来就接近正交）。图好看，但读不出信息，等于装饰。

换成**混淆画像**：
  拿每一类的 N 张真实样本跑一遍，取 softmax 的平均，得到
      M[真值类][它猜的类] = 平均概率
  再把 148 行当 148 个向量做 PCA。
于是"挨得近"不再是"权重方向相似"这种抽象说法，而是**这个网络会把这两个词
认错**。地图上有邻居，背后就是一条真实会发生的误判。

产物 `assets/model/classmap.json`：每类一个 (x, y) + 它最容易混的前 3 名。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from export_model import ASSET_DIR, OUT, DoodleNetSep  # noqa: E402
from make_grad_vectors import build_dequant_ops  # noqa: E402

PER_CLASS = 24


def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    ck = torch.load(OUT / "doodlenet_sep.pt", map_location="cpu", weights_only=False)
    words = ck["words"]
    n = len(words)
    model = DoodleNetSep(n, width=tuple(ck["width"]))
    model.load_state_dict(ck["state"])
    model.eval()
    ops = build_dequant_ops(model)

    hw = model.head.weight.detach().numpy().astype(np.float32)
    hb = model.head.bias.detach().numpy().astype(np.float32)
    from export_model import quantize_weight

    hq, hscale = quantize_weight(hw[:, None, :])
    hw_deq = hq.reshape(hw.shape).astype(np.float32) * hscale[:, None]

    d = np.load(OUT / "doodles.npz", allow_pickle=True)
    X = d["X"]
    y = d["y"].astype(np.int64)
    idx = []
    for c in range(n):
        where = np.flatnonzero(y == c)
        idx.extend(where[-PER_CLASS:].tolist())  # 取每类尾部样本，训练时它们排序靠前，基本没参加训练
    idx = np.asarray(idx)
    print(f"用 {len(idx)} 张图算混淆画像（每类 {PER_CLASS} 张）", flush=True)

    xb = torch.from_numpy(X[idx].astype(np.float32).reshape(-1, 1, 32, 32) / 255.0)
    with torch.no_grad():
        h = xb
        for op in ops:
            w = torch.from_numpy(op["w"])
            b = torch.from_numpy(op["b"])
            if op["type"] == 0:
                h = torch.nn.functional.conv2d(h, w, b, padding=1)
            elif op["type"] == 1:
                h = torch.nn.functional.conv2d(h, w, b, padding=1, groups=op["cin"])
            else:
                h = torch.nn.functional.conv2d(h, w, b)
            h = torch.nn.functional.relu(h)
            if op["pool"]:
                h = torch.nn.functional.max_pool2d(h, 2)
        gap = h.mean(dim=(2, 3))
        logits = (gap @ torch.from_numpy(hw_deq).T + torch.from_numpy(hb)).numpy()

    probs = softmax_np(logits)
    truth = y[idx]

    # 混淆画像：每行是该类样本的平均预测分布
    M = np.zeros((n, n), dtype=np.float64)
    for c in range(n):
        sel = probs[truth == c]
        if len(sel) == 0:
            continue
        M[c] = sel.mean(axis=0)

    # 每类最容易混的前三名（排除自己）
    mixed: dict[int, list[int]] = {}
    for c in range(n):
        row = M[c].copy()
        row[c] = -1
        mixed[c] = np.argsort(-row)[:3].tolist()

    acc = float((probs.argmax(axis=1) == truth).mean())
    print(f"这批样本的 top-1 准确率 {acc * 100:.1f}%", flush=True)

    # ── 布局 ──
    # 试过两种都不行：
    #   PCA：混淆画像几乎是均匀的（每类的概率摊在 148 类上），前两个主成分只解释
    #        2.4% 的方差，投出来是一团噪声。
    #   MDS：同样原因，二维坐标复现原始距离的相关性只有 6% —— 这套混淆关系本来
    #        就是高维的，硬压成平面只会失真。
    # 换成**稀疏图 + 力导向布局**：只保留每类最容易混的 3 条边（它们合起来就是
    # 玩家会撞见的误判），再用 Fruchterman-Reingold 把有边的节点拉到一起、没边的
    # 推开。力导向不声称保持全局距离，它只负责"连着的聚在一起"，而这正是要表达的。
    K = 3
    edges: list[tuple[int, int, float]] = []
    for c in range(n):
        row = M[c].copy()
        row[c] = -1.0
        top = np.argsort(-row)[:K]
        for j in top:
            w = float(row[j])
            if w <= 1e-6:
                continue
            a, b = (c, int(j)) if c < j else (int(j), c)
            edges.append((a, b, w))
    # 合并重复边（a→b 和 b→a 都可能是前 K 名）
    merged: dict[tuple[int, int], float] = {}
    for a, b, w in edges:
        key = (a, b)
        merged[key] = max(merged.get(key, 0.0), w)
    edges = [(a, b, w) for (a, b), w in merged.items()]
    wmax = max(w for _, _, w in edges) if edges else 1.0
    edges = [(a, b, w / wmax) for a, b, w in edges]
    print(f"混淆图：{len(edges)} 条边", flush=True)

    # Fruchterman-Reingold（固定种子，结果可复现；不需要 pip 依赖）
    rng = np.random.default_rng(20260118)
    pos = rng.random((n, 2)) * 0.9 + 0.05
    area = 1.0
    k_ideal = np.sqrt(area / n) * 0.55
    adj = [[] for _ in range(n)]
    for a, b, w in edges:
        adj[a].append((b, w))
        adj[b].append((a, w))

    for it in range(420):
        temp = 0.09 * (1 - it / 420) + 0.004
        disp = np.zeros((n, 2))
        # 斥力
        for i in range(n):
            d = pos[i] - pos
            dist = np.maximum(1e-4, np.linalg.norm(d, axis=1))
            f = (k_ideal ** 2) / dist
            f[i] = 0.0
            disp[i] += (d / dist[:, None] * f[:, None]).sum(axis=0)
        # 引力（只沿边，按混淆强度加权）
        for i in range(n):
            for j, w in adj[i]:
                d = pos[i] - pos[j]
                dist = max(1e-4, float(np.linalg.norm(d)))
                f = (dist ** 2) / k_ideal * (0.4 + 0.6 * w)
                disp[i] -= d / dist * f
        # 限制步长
        len_ = np.maximum(1e-4, np.linalg.norm(disp, axis=1))
        pos += disp / len_[:, None] * np.minimum(len_, temp)[:, None]
        pos = np.clip(pos, 0.02, 0.98)

    xy = pos
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(1e-8, hi - lo)
    xy = (xy - lo) / span * 0.92 + 0.04

    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(1e-8, hi - lo)
    xy = (xy - lo) / span * 0.92 + 0.04

    payload = json.loads((ASSET_DIR / "doodlemind-v4.json").read_text(encoding="utf-8"))
    by_en = {c["en"]: c for c in payload["classes"]}
    points = []
    for i, en in enumerate(words):
        c = by_en.get(en, {})
        points.append(
            {
                "en": en,
                "zh": c.get("zh", en),
                "emoji": c.get("emoji", ""),
                "x": round(float(xy[i, 0]), 4),
                "y": round(float(xy[i, 1]), 4),
                "mixed": [words[j] for j in mixed[i]],
            }
        )

    path = ASSET_DIR / "classmap.json"
    path.write_text(
        json.dumps(
            {
                "size": 2,
                "points": points,
                # 边的两端是"它会认错的一对"，w 是归一化后的混淆强度 0..1
                "edges": [{"a": a, "b": b, "w": round(w, 3)} for a, b, w in edges],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"写出 {path} ({path.stat().st_size} 字节, {n} 类, {len(edges)} 条边)", flush=True)

    names = ["猫", "太阳", "长颈鹿", "房子"]
    by_zh = {p["zh"]: p for p in points}
    for name in names:
        p = by_zh.get(name)
        if p:
            print(f"  {name} 最常被认成：{p['mixed']}", flush=True)


if __name__ == "__main__":
    main()
