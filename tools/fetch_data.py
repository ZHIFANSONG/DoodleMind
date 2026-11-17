"""从 Google QuickDraw 拉取简化笔画数据，栅格化成 32x32 训练集。

数据源：https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson
每行一个样本：{"word","countrycode","timestamp","recognized","key_id","drawing":[[xs],[ys]]}

关键点：**边下边解析边截断**。每类文件 70MB+，但我们每类只要几千条，
读够就断开连接，所以实际下载量只有几百 MB。

产出的栅格化结果必须与 `src/lib/raster.ts` 数学等价，否则线上推理会串味。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from categories import EN_NAMES  # noqa: E402

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "tools" / "cache"
OUT = ROOT / "tools" / "out"

SIZE = 32
PAD = 3.0
RADIUS = 1.0


def rasterize_np(strokes: list[list[list[float]]], size: int = SIZE, pad: float = PAD, radius: float = RADIUS) -> np.ndarray:
    """与 raster.py::rasterize 数学等价的向量化实现。"""
    canvas = np.zeros((size, size), dtype=np.float32)

    pts_x: list[float] = []
    pts_y: list[float] = []
    for s in strokes:
        if len(s) < 2 or len(s[0]) < 1:
            continue
        pts_x.extend(s[0])
        pts_y.extend(s[1])
    if not pts_x:
        return canvas

    min_x, max_x = min(pts_x), max(pts_x)
    min_y, max_y = min(pts_y), max(pts_y)
    w = max_x - min_x
    h = max_y - min_y
    span = max(w, h, 1e-6)
    scale = (size - 2.0 * pad) / span
    off_x = pad + ((size - 2.0 * pad) - w * scale) / 2.0
    off_y = pad + ((size - 2.0 * pad) - h * scale) / 2.0

    segs = []
    for s in strokes:
        if len(s) < 2 or len(s[0]) < 1:
            continue
        sx = np.asarray(s[0], dtype=np.float64) * scale + off_x
        sy = np.asarray(s[1], dtype=np.float64) * scale + off_y
        if len(sx) == 1:
            sx = np.array([sx[0], sx[0]], dtype=np.float64)
            sy = np.array([sy[0], sy[0]], dtype=np.float64)
        segs.append(np.stack([sx[:-1], sy[:-1], sx[1:], sy[1:]], axis=1))
    if not segs:
        return canvas
    seg = np.concatenate(segs, axis=0)  # (S, 4)

    gx = np.arange(size, dtype=np.float64) + 0.5
    gy = np.arange(size, dtype=np.float64) + 0.5
    px = gx[None, :]           # (1, size)
    py = gy[:, None]           # (size, 1)
    px = np.broadcast_to(px, (size, size))
    py = np.broadcast_to(py, (size, size))

    ax = seg[:, 0][:, None, None]
    ay = seg[:, 1][:, None, None]
    bx = seg[:, 2][:, None, None]
    by = seg[:, 3][:, None, None]

    dx = bx - ax
    dy = by - ay
    vx = px[None, :, :] - ax
    vy = py[None, :, :] - ay
    denom = dx * dx + dy * dy
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    t = np.clip((vx * dx + vy * dy) / denom, 0.0, 1.0)
    ex = vx - t * dx
    ey = vy - t * dy
    dist = np.sqrt(ex * ex + ey * ey)          # (S, size, size)

    cov = np.clip(radius + 0.5 - dist, 0.0, 1.0)
    canvas = cov.max(axis=0).astype(np.float32)
    return canvas


def sample_to_u8(sample: dict) -> np.ndarray | None:
    if not sample.get("recognized", True):
        return None
    strokes = sample.get("drawing")
    if not strokes:
        return None
    total_pts = sum(len(s[0]) for s in strokes if len(s) >= 2)
    if total_pts < 3:
        return None
    img = rasterize_np(strokes)
    ink = float(img.sum())
    if ink < 4.0:  # 几乎没画出东西（全挤在一两个像素）→ 丢掉
        return None
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def fetch_class(word: str, limit: int, timeout: float = 60.0) -> np.ndarray:
    """流式读取某一类的 ndjson，够了就断连。返回 (N, SIZE*SIZE) uint8。"""
    cache_file = CACHE / f"{word.replace(' ', '_')}_{limit}.npy"
    if cache_file.exists():
        arr = np.load(cache_file)
        if arr.shape[0] >= limit:
            return arr

    url = BASE_URL.format(urllib.request.quote(word))
    got: list[np.ndarray] = []
    seen = 0
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "doodlemind-dataset/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        reader = io.BufferedReader(resp, buffer_size=1 << 20)
        for raw in reader:
            seen += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            img = sample_to_u8(obj)
            if img is None:
                continue
            got.append(img)
            if len(got) >= limit:
                break
    elapsed = time.time() - t0
    arr = np.stack(got, axis=0) if got else np.zeros((0, SIZE * SIZE), dtype=np.uint8)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, arr)
    print(f"  {word:<20} kept {len(got):>5}/{seen:<6} in {elapsed:5.1f}s", flush=True)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=4000, help="每类保留多少样本")
    ap.add_argument("--classes", type=int, default=0, help="只处理前 N 类（调试用）")
    args = ap.parse_args()

    words = EN_NAMES[: args.classes] if args.classes else EN_NAMES
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    X_parts = []
    y_parts = []
    t_start = time.time()
    for i, word in enumerate(words):
        print(f"[{i + 1:>3}/{len(words)}] {word}", flush=True)
        arr = fetch_class(word, args.per_class)
        if arr.shape[0] == 0:
            print(f"  !! {word} 全被过滤掉了，检查数据源", flush=True)
            continue
        X_parts.append(arr)
        y_parts.append(np.full(arr.shape[0], i, dtype=np.uint8))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    perm = np.random.default_rng(20261117).permutation(X.shape[0])
    X = X[perm]
    y = y[perm]

    np.savez_compressed(OUT / "doodles.npz", X=X, y=y, words=np.array(words))
    print(
        f"\n完成：X={X.shape} y={y.shape} 类别={len(words)} "
        f"用时={time.time() - t_start:.0f}s -> {OUT / 'doodles.npz'}",
        flush=True,
    )
    print("每类样本数：", {w: int((y == i).sum()) for i, w in enumerate(words[:5])}, "...", flush=True)


if __name__ == "__main__":
    main()
