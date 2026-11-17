"""评估「边画边判」到底有没有变强。

为什么需要这个脚本：
    常规 top-1 是在**画完的图**上算的。但我们真正的场景是玩家还在画的时候。
    一个模型完全可能在完整图上是 81%、在「只画了一半」的图上是 30%——那它在真机
    上的体感就是「一直不说话，松手才跳」，哪怕离线指标很漂亮。

    所以按墨迹比例做三档截断（保留 25% / 50% / 75% 的总弧长），分别算 top-1。
    只有这三档一起看，才知道模型是真的「越画越确定」，还是「等齐了才表态」。

数据从哪来：
    `doodles.npz` 存的是栅格化后的 32×32 图，没法反过来截断。所以这里按需从
    QuickDraw simplified 现抓少量**原始笔画**并缓存到 `tools/cache/raw/`——
    每类只读够 N 条就断连，148 类合计几 MB，可重复跑。

几何口径：
    直接复用 `tools/fetch_data.py` 的 `rasterize_np`（与 `tools/raster.py::rasterize`
    数学等价，但向量化），不在这里另抄一份。口径一旦分家，两个脚本会给出互相矛盾的
    结论，而矛盾的那一对里至少有一个是错的。

用法：
    python tools/eval_partial.py --ckpt doodlenet_big.pt
    python tools/eval_partial.py --ckpt doodlenet_big.pt --compare doodlenet.pt
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import urllib.request

import numpy as np
import torch

from fetch_data import rasterize_np
from train import DoodleNet, DoodleNetSep, OUT, CATEGORIES

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
RAW_CACHE = OUT.parent / "cache" / "raw"
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"


def fetch_raw(word: str, per_class: int) -> list[list[list[float]]]:
    """抓某一类的原始笔画（只读够就断连），带磁盘缓存。"""
    path = RAW_CACHE / f"{word.replace(' ', '_')}_{per_class}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())

    url = BASE_URL.format(urllib.request.quote(word))
    out: list[list[list[float]]] = []
    req = urllib.request.Request(url, headers={"User-Agent": "doodlemind-eval/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in io.BufferedReader(resp, buffer_size=1 << 20):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not obj.get("recognized", True):
                continue
            d = obj.get("drawing")
            if not isinstance(d, list) or len(d) < 2:
                continue
            if sum(len(s) for s in d) < 3:
                continue
            out.append(d)
            if len(out) >= per_class:
                break

    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def truncate(strokes: list[list[list[float]]], frac: float) -> list[list[list[float]]]:
    """按总弧长截断笔画。frac>=1.0 时原样返回。"""
    if frac >= 1.0:
        return strokes

    lengths = []
    total = 0.0
    for s in strokes:
        xs, ys = s[0], s[1]
        seg = sum(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(len(xs) - 1))
        lengths.append(seg)
        total += seg
    if total <= 0:
        return strokes

    budget = total * frac
    kept: list[list[list[float]]] = []
    for s, seg in zip(strokes, lengths):
        if budget <= 0:
            break
        if seg <= budget:
            kept.append(s)
            budget -= seg
            continue
        xs, ys = s[0], s[1]
        acc = 0.0
        k = 0
        for i in range(len(xs) - 1):
            step = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            if acc + step > budget:
                break
            acc += step
            k = i + 1
        if k >= 1:
            kept.append([xs[: k + 1], ys[: k + 1]])
        budget = 0
    return kept or strokes


def build_images(drawings: list, frac: float) -> np.ndarray:
    imgs = []
    for d in drawings:
        img = rasterize_np(truncate(d, frac))
        imgs.append(np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8))
    return np.stack(imgs, axis=0)


def load_ckpt(path: str):
    ck = torch.load(OUT / path, map_location="cpu", weights_only=False)
    # 结构跟着 checkpoint 走：dense 与 sep 的参数量、层名都不一样，
    # 拿错类去 load_state_dict 会报一堆 key 不匹配，或者更糟——什么都不报但权重全乱。
    arch = ck.get("arch", "dense")
    cls = DoodleNetSep if arch == "sep" else DoodleNet
    model = cls(len(ck["words"]), width=tuple(ck["width"]))
    model.load_state_dict(ck["state"])
    model.eval()
    return model


def evaluate(ck_path: str, drawings: list, labels: np.ndarray) -> dict:
    model = load_ckpt(ck_path)
    res: dict = {}
    for frac in FRACTIONS:
        X = build_images(drawings, frac).astype(np.float32) / 255.0
        preds = []
        logits_all = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                lg = model(torch.from_numpy(X[i : i + 512]).unsqueeze(1))
                logits_all.append(lg.numpy())
                preds.append(lg.argmax(dim=1).numpy())
        pred = np.concatenate(preds)
        logits = np.concatenate(logits_all, axis=0)

        # 光看 top-1 是不够的：25% 弧长时人自己也认不出画的是什么，top-1 必然极低，
        # 但那不代表体验差。玩家真正在意的是「候选栏前 6 个里有没有对的词」以及
        # 「对的词排第几」——判读区显示的正是排名，不是置信度本身。
        order = np.argsort(-logits, axis=1)  # (n, nClasses)，降序
        ranks = np.array([int(np.where(order[i] == labels[i])[0][0]) for i in range(len(labels))])

        k = f"{int(frac * 100)}%"
        res[k] = {
            "top1": float((pred == labels).mean()),
            "top5": float((ranks < 5).mean()),
            "top1_candidates": float((ranks < 6).mean()),  # 候选栏正好 6 个
            "mrr": float((1.0 / (ranks + 1)).mean()),
            "median_rank": int(np.median(ranks)),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="doodlenet_big.pt")
    ap.add_argument("--compare", type=str, default="", help="另一个 checkpoint，用于对照")
    ap.add_argument("--per-class", type=int, default=10, help="每类抓多少张原始样本")
    args = ap.parse_args()

    drawings: list = []
    labels: list[int] = []
    for ci, (en, _zh, _emoji) in enumerate(CATEGORIES):
        try:
            got = fetch_raw(en, args.per_class)
        except Exception as e:  # 网络问题不该让整轮评估白跑
            print(f"  !! {en}: {e}", flush=True)
            continue
        drawings.extend(got)
        labels.extend([ci] * len(got))
        if (ci + 1) % 20 == 0:
            print(f"  数据 {ci + 1}/{len(CATEGORIES)} 类，累计 {len(drawings)} 张", flush=True)

    y = np.array(labels, dtype=np.int64)
    print(f"\n评估 {len(drawings)} 张（每类最多 {args.per_class} 张，全部是没参与训练的原始样本）\n", flush=True)

    rows = {args.ckpt: evaluate(args.ckpt, drawings, y)}
    if args.compare:
        rows[args.compare] = evaluate(args.compare, drawings, y)

    for metric, title, fmt in (
        ("top1", "top-1（画完才重要）", "{:>6.3f}"),
        ("top1_candidates", "前 6 命中（候选栏里有对的）", "{:>6.3f}"),
        ("mrr", "MRR（越大越早排到前面）", "{:>6.3f}"),
        ("median_rank", "排名中位数（越小越好）", "{:>6d}"),
    ):
        head = "  ".join(f"{int(f * 100):>5}%" for f in FRACTIONS)
        print(f"\n{title}")
        print(f"{'checkpoint':<24} {head}")
        for name, r in rows.items():
            cells = "  ".join(fmt.format(r[f"{int(f * 100)}%"][metric]) for f in FRACTIONS)
            print(f"{name:<24} {cells}")

    if args.compare:
        a, b = rows[args.ckpt], rows[args.compare]
        print("\n增益（左减右，正数=更好）：")
        for f in FRACTIONS:
            k = f"{int(f * 100)}%"
            print(
                f"  {k:>5}: top1 {a[k]['top1'] - b[k]['top1']:+.4f}"
                f"  前6 {a[k]['top1_candidates'] - b[k]['top1_candidates']:+.4f}"
                f"  MRR {a[k]['mrr'] - b[k]['mrr']:+.4f}"
            )

    (OUT / "eval_partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n结果写入 {OUT / 'eval_partial.json'}", flush=True)


if __name__ == "__main__":
    main()
