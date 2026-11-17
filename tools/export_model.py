"""把训练好的 DoodleNet 量化导出成端上直接可读的模型文件。

量化方案（关键决策）
------------------
卷积权重：**per-output-channel 对称 int8**（-127..127），wScale = max(|w|)/127。
激活：**不做量化**，端上是 float32。

为什么不做全 int8：JS 里没有 SIMD，整数和浮点的循环开销一样；而激活量化会在
4 段 conv 里逐层累积误差，top-1 会掉好几个点。权重压到 int8 已经把资源体积
从 650KB 降到 163KB，运行时解量化成 Float32Array 只花 650KB 内存——这是最划算的
取舍。BN 全部折叠进 conv 的权重和偏置，推理时不存在 BN 层。

输出（全部小端）
--------------
magic    4B  "DDMN"
version  u8  = 2
nClasses u16
nBlocks  u16
每块：
  cin    u16
  cout   u16
  wScale f32[cout]
  bias   f32[cout]     已折叠 BN
  w      int8[cout][cin][3][3]   行主序
分类头：
  width  u16
  headW  f32[nClasses*width]
  headB  f32[nClasses]
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from categories import CATEGORIES  # noqa: E402
from train import DoodleNet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tools" / "out"
ASSET_DIR = ROOT / "assets" / "model"


def fold_bn(block) -> tuple[np.ndarray, np.ndarray]:
    """返回折叠 BN 后的 (weight (cout,cin,3,3), bias (cout,))。"""
    w = block.conv.weight.detach().numpy().astype(np.float64)
    bn = block.bn
    gamma = bn.weight.detach().numpy().astype(np.float64)
    beta = bn.bias.detach().numpy().astype(np.float64)
    mean = bn.running_mean.detach().numpy().astype(np.float64)
    var = bn.running_var.detach().numpy().astype(np.float64)
    k = gamma / np.sqrt(var + float(bn.eps))
    return w * k[:, None, None, None], beta - mean * k


def quantize_weight(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cout = w.shape[0]
    flat = w.reshape(cout, -1)
    amax = np.abs(flat).max(axis=1)
    amax = np.where(amax < 1e-8, 1e-8, amax)
    scale = (amax / 127.0).astype(np.float32)
    q = np.clip(np.rint(flat / scale[:, None]), -127, 127).astype(np.int8)
    return q.reshape(w.shape), scale


def conv3x3_np(x: np.ndarray, w: np.ndarray, b: np.ndarray, relu: bool = True) -> np.ndarray:
    """朴素 3x3 conv，只用于离线校验（x: (cin,h,w), w: (cout,cin,3,3)）。"""
    cin, h, wd = x.shape
    cout = w.shape[0]
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((cout, h, wd), dtype=np.float32)
    for co in range(cout):
        acc = np.zeros((h, wd), dtype=np.float32)
        for ci in range(cin):
            k = w[co, ci]
            for ky in range(3):
                for kx in range(3):
                    acc += k[ky, kx] * p[ci, ky : ky + h, kx : kx + wd]
        acc += b[co]
        out[co] = np.maximum(acc, 0.0) if relu else acc
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="doodlenet.pt")
    ap.add_argument("--name", type=str, default="doodlemind-v1")
    args = ap.parse_args()

    ck = torch.load(OUT / args.ckpt, map_location="cpu", weights_only=False)
    width = tuple(ck["width"])
    words = ck["words"]
    model = DoodleNet(len(words), width=width)
    model.load_state_dict(ck["state"])
    model.eval()

    d = np.load(OUT / "doodles.npz", allow_pickle=True)
    Xall = d["X"].astype(np.float32) / 255.0
    yall = d["y"].astype(np.int64)

    blocks = [model.b1, model.b2, model.b3, model.b4]
    buf = bytearray(b"DDMN" + struct.pack("<BHH", 2, len(words), len(blocks)))

    folded = []
    for bi, blk in enumerate(blocks):
        w, b = fold_bn(blk)
        qw, wscale = quantize_weight(w)
        cin, cout = w.shape[1], w.shape[0]
        buf += struct.pack("<HH", cin, cout)
        buf += wscale.astype(np.float32).tobytes()
        buf += b.astype(np.float32).tobytes()
        buf += qw.astype(np.int8).tobytes(order="C")
        w_f32 = qw.astype(np.float32) * wscale[:, None, None, None]  # 解量化后的等效权重
        folded.append(w_f32.astype(np.float32))
        print(f"  block{bi + 1}: {cin}->{cout}  int8 {qw.size / 1024:6.1f} KB  wScale[0..2]={wscale[:3]}", flush=True)

    hw = model.head.weight.detach().numpy().astype(np.float32)
    hb = model.head.bias.detach().numpy().astype(np.float32)
    buf += struct.pack("<H", hw.shape[1])
    buf += hw.tobytes(order="C")
    buf += hb.tobytes(order="C")

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    bin_name = f"{args.name}.bin"
    (ASSET_DIR / bin_name).write_bytes(bytes(buf))
    print(f"\n二进制 {ASSET_DIR / bin_name}  {len(buf) / 1024:.1f} KB", flush=True)

    # ---- 导出 JS 端直接吃的数据：类别 + base64 权重 ----
    b64 = base64.b64encode(bytes(buf)).decode("ascii")
    payload = {
        "format": 2,
        "name": args.name,
        "bytes": len(buf),
        "nClasses": len(words),
        "width": list(width),
        "classes": [
            {"en": CATEGORIES[i][0], "zh": CATEGORIES[i][1], "emoji": CATEGORIES[i][2]} for i in range(len(words))
        ],
        "weights": b64,
    }
    json_path = ASSET_DIR / f"{args.name}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"JS 数据 {json_path}  {json_path.stat().st_size / 1024:.1f} KB", flush=True)

    # ---- 用「解量化等效权重 + fp32」重算，验证 int8 精度损失可接受 ----
    n_test = 1500
    Xt = Xall[:n_test]
    yt = yall[:n_test]

    biases = []
    for blk in blocks:
        _, b = fold_bn(blk)
        biases.append(b.astype(np.float32))

    def np_forward(img: np.ndarray) -> np.ndarray:
        x = img[None, :, :].astype(np.float32)
        for bi in range(len(folded)):
            x = conv3x3_np(x, folded[bi], biases[bi], relu=True)
            if bi < 3:
                c, h, wd = x.shape
                x = x.reshape(c, h // 2, 2, wd // 2, 2).max(axis=(2, 4))
        gap = x.reshape(x.shape[0], -1).mean(axis=1)
        return (hw @ gap + hb).astype(np.float32)

    with torch.no_grad():
        ref = model(torch.from_numpy(Xt).unsqueeze(1)).numpy()
    ref_pred = ref.argmax(axis=1)
    sub = np.arange(0, n_test, 3)  # 朴素 conv 慢，抽样 500 张
    agree = 0
    for i in sub:
        if int(np_forward(Xt[i]).argmax()) == int(ref_pred[i]):
            agree += 1
    print(
        f"int8→fp32 与原始 fp32 的 top-1 一致率：{agree}/{len(sub)} = {agree / len(sub):.4f}",
        flush=True,
    )
    print(f"原始 fp32 在这 {n_test} 张上 top1 = {(ref_pred == yt).mean():.4f}", flush=True)

    # ---- 跨语言验证向量：给 Node 侧跑 TS 引擎对比 ----
    n_vec = 120
    vec = {
        "images": (Xall[:n_vec] * 255.0).round().astype(np.uint8).reshape(n_vec, -1).tolist(),
        "labels": yt[:n_vec].tolist(),
        "refPred": ref_pred[:n_vec].tolist(),
        "refTop5": [[int(v) for v in row] for row in np.argsort(-ref[:n_vec], axis=1)[:, :5].tolist()],
    }
    (OUT / "test_vectors.json").write_text(json.dumps(vec), encoding="utf-8")
    print(f"验证向量 {OUT / 'test_vectors.json'}", flush=True)


if __name__ == "__main__":
    main()
