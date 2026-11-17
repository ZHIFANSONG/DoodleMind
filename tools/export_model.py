"""把训练好的 DoodleNet 量化导出成端上直接可读的模型文件。

量化方案（关键决策）
------------------
卷积权重：**per-output-channel 对称 int8**（-127..127），wScale = max(|w|)/127。
分类头：  v3 起也做 **per-class 对称 int8**（每行一组 scale）。
激活：    **不做量化**，端上是 float32。

为什么不做全 int8：JS 里没有 SIMD，整数和浮点的循环开销一样；而激活量化会在
各段 conv 里逐层累积误差，top-1 会掉好几个点。权重压到 int8 已经把资源体积
从 1.2MB 降到 200KB 上下，运行时解量化成 Float32Array 也只占几百 KB——这是最划算的
取舍。BN 全部折叠进 conv 的权重和偏置，推理时不存在 BN 层。

为什么 head 量化不伤精度：GAP 之后的输入是各通道均值，非负、量级集中（0..~3），
而 head 权重每行的动态范围差异是主要误差源——per-class 的 scale 正好吃掉它。
导出时会打印 int8 head 与 fp32 head 对照真值的准确率差，掉超过 0.3 个百分点就
说明不该量化。

v4：为什么模型格式要带「算子类型」
--------------------------------
`train.py --arch sep` 的网络是 depthwise 3x3 → pointwise 1x1，和原来的 dense 3x3
混合出现（第一层仍是 dense，因为输入只有 1 通道）。三种算子的权重形状完全不同：

    dense 3x3   w[cout][cin][3][3]     cout*cin*9
    dw    3x3   w[cin][1][3][3]        cin*9
    pw    1x1   w[cout][cin][1][1]     cout*cin

所以每个 op 前面都写一个类型字节，端上按类型分派。硬编码「4 个 block 全是 dense」
的话，换结构就得改两端的解析代码，而两端的解析代码一旦不同步，闸门会静默放行。

输出（全部小端）
--------------
magic    4B  "DDMN"
version  u8  = 4
nClasses u16
nOps     u16
每个 op：
  type   u8   0=dense3x3  1=depthwise3x3  2=pointwise1x1
  flags  u8   bit0 = 该 op 之后接 2x2 maxpool
  cin    u16
  cout   u16
  wScale f32[cout]
  bias   f32[cout]         已折叠 BN
  w      int8[count]       count 由 type 决定，见上
分类头：
  width     u16
  headScale f32[nClasses]
  headW     int8[nClasses*width]
  headB     f32[nClasses]
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
from train import DoodleNet, DoodleNetSep  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tools" / "out"
ASSET_DIR = ROOT / "assets" / "model"
# 验证向量必须入库：CI 要在没有 Python 的机器上跑 check-engine.mjs。
# 放 tools/out/ 会被 gitignore 掉，闸门在 CI 里就成了空跑。
VEC_DIR = ROOT / "tools" / "vectors"

T_DENSE = 0
T_DW = 1
T_PW = 2
FLAG_POOL = 1


def fold_bn(conv, bn) -> tuple[np.ndarray, np.ndarray]:
    """把 BN 折进 conv：返回 (weight, bias)。

    conv 之后紧接 BN 时，推理等价于一个带偏置的卷积：
        y = γ(x−μ)/√(σ²+ε) + β
          = (γ/√(σ²+ε))·x + (β − μγ/√(σ²+ε))
    所以 w' = w·k，b' = β − μ·k，k = γ/√(σ²+ε)。

    对 depthwise 也成立：它的 weight 是 (cin,1,3,3)，按通道广播的 `k[:,None,None,None]`
    正好对得上，不需要区分 groups。
    """
    w = conv.weight.detach().numpy().astype(np.float32)
    gamma = bn.weight.detach().numpy().astype(np.float32)
    beta = bn.bias.detach().numpy().astype(np.float32)
    mean = bn.running_mean.detach().numpy().astype(np.float32)
    var = bn.running_var.detach().numpy().astype(np.float32)
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


def weight_count(kind: int, cin: int, cout: int) -> int:
    if kind == T_DENSE:
        return cout * cin * 9
    if kind == T_DW:
        return cin * 9
    return cout * cin


# ---------------------------------------------------------------- 朴素 numpy 参考
# 只用于离线自检「同一套 int8 权重，numpy 与 torch 结果是否一致」。
# 刻意写成最直白的形式：它越笨，越容易看出是解析/布局出了问题。

def conv3x3_np(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    cin, h, wd = x.shape
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.empty((w.shape[0], h, wd), dtype=np.float32)
    for co in range(w.shape[0]):
        acc = np.zeros((h, wd), dtype=np.float32)
        for ci in range(cin):
            k = w[co, ci]
            for ky in range(3):
                for kx in range(3):
                    acc += k[ky, kx] * p[ci, ky : ky + h, kx : kx + wd]
        out[co] = np.maximum(acc + b[co], 0.0)
    return out


def dw3x3_np(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    c, h, wd = x.shape
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.empty((c, h, wd), dtype=np.float32)
    for ci in range(c):
        acc = np.zeros((h, wd), dtype=np.float32)
        k = w[ci, 0]
        for ky in range(3):
            for kx in range(3):
                acc += k[ky, kx] * p[ci, ky : ky + h, kx : kx + wd]
        out[ci] = np.maximum(acc + b[ci], 0.0)
    return out


def pw1x1_np(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    cin, h, wd = x.shape
    flat = x.reshape(cin, -1)
    out = (w[:, :, 0, 0] @ flat) + b[:, None]
    return np.maximum(out, 0.0).reshape(w.shape[0], h, wd)


def reference_logits(x: torch.Tensor, ops: list[dict], head_w: np.ndarray, head_b: np.ndarray) -> torch.Tensor:
    """用**解量化后的 int8 权重**重跑一遍前向，作为跨语言校验的参考。

    为什么参考实现不能用训练时的原始 fp32 模型：
    量化本身就会让 top-1 有 ~1% 的抖动。如果拿原始 fp32 当参考，Node 侧的校验就
    同时混进了「量化误差」和「实现 bug」两个变量，阈值定多少都不对——定松了抓不到
    真 bug，定紧了永远红。

    所以参考端也用解量化权重，这样两边跑的是**数学上同一个网络**，
    剩下的任何不一致都只可能来自 TS 实现本身。
    """
    import torch.nn.functional as F  # noqa: N812

    h = x
    for op in ops:
        w = torch.from_numpy(op["w"])
        b = torch.from_numpy(op["b"])
        if op["type"] == T_DENSE:
            h = F.conv2d(h, w, b, padding=1)
        elif op["type"] == T_DW:
            h = F.conv2d(h, w, b, padding=1, groups=op["cin"])
        else:
            h = F.conv2d(h, w, b)
        h = F.relu(h)
        if op["pool"]:
            h = F.max_pool2d(h, 2)
    gap = h.mean(dim=(2, 3))
    return gap @ torch.from_numpy(head_w).T + torch.from_numpy(head_b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="doodlenet.pt")
    ap.add_argument("--name", type=str, default="doodlemind-v1")
    args = ap.parse_args()

    ck = torch.load(OUT / args.ckpt, map_location="cpu", weights_only=False)
    width = tuple(ck["width"])
    words = ck["words"]
    # arch 从 checkpoint 里读，不从命令行读：否则 `--arch sep` 配 dense 的 ckpt
    # 会在 load_state_dict 那里报一堆 key 不匹配，或者更糟——静默按 dense 导出。
    arch = ck.get("arch", "dense")
    net_cls = DoodleNetSep if arch == "sep" else DoodleNet
    model = net_cls(len(words), width=width)
    model.load_state_dict(ck["state"])
    model.eval()
    print(f"结构 {arch}  width={width}", flush=True)

    d = np.load(OUT / "doodles.npz", allow_pickle=True)
    Xall = d["X"].astype(np.float32) / 255.0
    yall = d["y"].astype(np.int64)

    # 展开成线性 op 列表。pool 只在每段的最后一个 op 后面挂一次，
    # 否则一个 sep 段会连池化两次，特征图直接塌掉。
    ops: list[dict] = []
    if arch == "sep":
        seq = [
            ("dense", model.b1),
            ("dw", model.b2), ("pw", model.b2),
            ("dw", model.b3), ("pw", model.b3),
            ("dw", model.b4), ("pw", model.b4),
        ]
    else:
        seq = [("dense", b) for b in (model.b1, model.b2, model.b3, model.b4)]

    for kind, blk in seq:
        if kind == "dense":
            w, b = fold_bn(blk.conv, blk.bn)
            t = T_DENSE
            # dense: (cout, cin, 3, 3)
            cin_, cout_ = w.shape[1], w.shape[0]
        elif kind == "dw":
            w, b = fold_bn(blk.dw, blk.dw_bn)
            t = T_DW
            # depthwise 的 weight 是 (c, 1, 3, 3)：groups=cin 时 torch 只存一份，
            # 所以 shape[0] 才是通道数，shape[1] 恒为 1。用 shape[1] 当 cin 会得到 1，
            # 于是 weight_count 算出 9 而不是 c*9 —— 导出时才 assert 炸掉。
            cin_ = cout_ = w.shape[0]
        else:
            w, b = fold_bn(blk.pw, blk.pw_bn)
            t = T_PW
            # pointwise: (cout, cin, 1, 1)
            cin_, cout_ = w.shape[1], w.shape[0]
        ops.append({"type": t, "cin": cin_, "cout": cout_, "w": w, "b": b, "pool": False})

    # 池化点：必须挂在「每段的最后一个 op」上。sep 版一段有两个 op（dw+pw），
    # 挂到 dw 上会让池化发生在通道混之前，特征图直接塌一半还多。
    #   dense: op0/op1/op2 之后各池一次，op3（4x4）不池
    #   sep:   seq = [dense(b1), dw2, pw2, dw3, pw3, dw4, pw4]
    #          32→16 在 b1 之后（op0），16→8 在 pw2 之后（op2），8→4 在 pw3 之后（op4）
    pool_after = {0, 2, 4} if arch == "sep" else {0, 1, 2}
    for i in pool_after:
        ops[i]["pool"] = True

    buf = bytearray(b"DDMN" + struct.pack("<BHH", 4, len(words), len(ops)))

    decq: list[dict] = []
    for i, op in enumerate(ops):
        qw, wscale = quantize_weight(op["w"])
        assert qw.size == weight_count(op["type"], op["cin"], op["cout"]), "权重个数与类型不符"
        buf += struct.pack("<BBHH", op["type"], FLAG_POOL if op["pool"] else 0, op["cin"], op["cout"])
        buf += wscale.astype(np.float32).tobytes()
        buf += op["b"].astype(np.float32).tobytes()
        buf += qw.astype(np.int8).tobytes(order="C")
        w_f32 = qw.astype(np.float32) * wscale.reshape(
            (-1,) + (1,) * (qw.ndim - 1)
        )  # 解量化后的等效权重
        decq.append({"type": op["type"], "cin": op["cin"], "cout": op["cout"], "w": w_f32, "b": op["b"], "pool": op["pool"]})
        name = {T_DENSE: "dense3x3", T_DW: "dw3x3", T_PW: "pw1x1"}[op["type"]]
        pool = " →pool" if op["pool"] else ""
        print(
            f"  op{i}: {name:9s} {op['cin']:>3}->{op['cout']:<3} int8 {qw.size / 1024:6.1f} KB{pool}",
            flush=True,
        )

    hw = model.head.weight.detach().numpy().astype(np.float32)
    hb = model.head.bias.detach().numpy().astype(np.float32)

    # head 也量化：per-class 对称 int8。借用同一函数：(N,1,W) 视作 N 个"通道"。
    hq, hscale = quantize_weight(hw[:, None, :])
    hq = hq.reshape(hw.shape)
    hw_deq = hq.astype(np.float32) * hscale[:, None]

    buf += struct.pack("<H", hw.shape[1])
    buf += hscale.astype(np.float32).tobytes()
    buf += hq.astype(np.int8).tobytes(order="C")
    buf += hb.tobytes(order="C")

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    bin_name = f"{args.name}.bin"
    (ASSET_DIR / bin_name).write_bytes(bytes(buf))
    print(f"\n二进制 {ASSET_DIR / bin_name}  {len(buf) / 1024:.1f} KB", flush=True)

    # ---- 导出 JS 端直接吃的数据：类别 + base64 权重 ----
    # 结构信息也写进来：关于页要显示参数量和网络结构，硬编码在 TSX 里的话
    # 每换一次模型就得手改一处文案，改了忘改就变成"界面上的数字是假的"。
    nparam = sum(int(p.numel()) for p in model.parameters())
    b64 = base64.b64encode(bytes(buf)).decode("ascii")
    payload = {
        "format": 4,
        "name": args.name,
        "arch": arch,
        "bytes": len(buf),
        "params": nparam,
        "nClasses": len(words),
        "width": list(width),
        "ops": [
            {
                "type": {T_DENSE: "dense3x3", T_DW: "dw3x3", T_PW: "pw1x1"}[op["type"]],
                "cin": op["cin"],
                "cout": op["cout"],
                "pool": op["pool"],
            }
            for op in ops
        ],
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

    def np_forward(img: np.ndarray) -> np.ndarray:
        x = img[None, :, :].astype(np.float32)
        for op in decq:
            if op["type"] == T_DENSE:
                x = conv3x3_np(x, op["w"], op["b"])
            elif op["type"] == T_DW:
                x = dw3x3_np(x, op["w"], op["b"])
            else:
                x = pw1x1_np(x, op["w"], op["b"])
            if op["pool"]:
                c, h, wd = x.shape
                x = x.reshape(c, h // 2, 2, wd // 2, 2).max(axis=(2, 4))
        gap = x.reshape(x.shape[0], -1).mean(axis=1)
        return (hw_deq @ gap + hb).astype(np.float32)

    with torch.no_grad():
        ref = reference_logits(torch.from_numpy(Xt).unsqueeze(1), decq, hw_deq, hb).numpy()
    ref_pred = ref.argmax(axis=1)

    # 朴素 numpy 很慢（没有向量化跨通道），sep 结构 op 数翻倍，抽样降到 150 张。
    sub = np.arange(0, n_test, max(1, n_test // 150))
    agree = sum(1 for i in sub if int(np_forward(Xt[i]).argmax()) == int(ref_pred[i]))
    print(
        f"朴素 numpy 与 torch(解量化权重) 的 top-1 一致率：{agree}/{len(sub)} = {agree / len(sub):.4f}",
        flush=True,
    )
    print(f"量化后 top1（对照真值）在这 {n_test} 张上 = {(ref_pred == yt).mean():.4f}", flush=True)
    with torch.no_grad():
        raw = model(torch.from_numpy(Xt).unsqueeze(1)).numpy()
    print(f"未量化 fp32 的 top1 = {(raw.argmax(axis=1) == yt).mean():.4f}", flush=True)

    # head 量化的单独代价：同一批图，只把分类头在 fp32 与 int8 之间切换。
    # 看的是**对照真值的准确率差**，不是两个头之间的预测一致率——
    # 两个头都猜错时也会"不一致"，但那种不一致对玩家没有任何影响。
    with torch.no_grad():
        ref_f32head = reference_logits(torch.from_numpy(Xt).unsqueeze(1), decq, hw, hb).numpy()
    acc_i8 = (ref_pred == yt).mean()
    acc_f32 = (ref_f32head.argmax(axis=1) == yt).mean()
    print(
        f"分类头 int8 {acc_i8:.4f} vs fp32 {acc_f32:.4f}"
        f"（差 {(acc_f32 - acc_i8) * 100:+.2f} 个点；掉超 0.3 就该放弃量化 head）",
        flush=True,
    )

    # ---- 跨语言验证向量：给 Node 侧跑 TS 引擎对比 ----
    # 除了 argmax，还导出**完整概率分布**。只比 argmax 是不够的：
    # 当两个类别概率接近时（比如 0.31 vs 0.30），实现里一个微小的偏差就会翻转排名，
    # 于是 96% 的一致率既可能是「哪都差一点点」，也可能是「某处真错了」。
    # 直接比概率数值能把这两者分开——正确实现应该一致到 1e-4 量级。
    n_vec = 120
    ref_probs = np.exp(ref[:n_vec] - ref[:n_vec].max(axis=1, keepdims=True))
    ref_probs /= ref_probs.sum(axis=1, keepdims=True)
    vec = {
        "images": (Xall[:n_vec] * 255.0).round().astype(np.uint8).reshape(n_vec, -1).tolist(),
        "labels": yt[:n_vec].tolist(),
        "refPred": ref_pred[:n_vec].tolist(),
        "refTop5": [[int(v) for v in row] for row in np.argsort(-ref[:n_vec], axis=1)[:, :5].tolist()],
        "refTop5Prob": [
            [round(float(p), 6) for p in row] for row in np.sort(-ref_probs, axis=1)[:, :5] * -1
        ],
    }
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    # 参考向量按模型名分开存：里面的 refPred / refTop5Prob 是用这一个模型算的，
    # 拿别的模型的参考去校验，一致率会掉到 70% 左右，看起来像引擎错了、实际是标尺错了。
    vec_path = VEC_DIR / f"test_vectors_{args.name}.json"
    vec_path.write_text(json.dumps(vec), encoding="utf-8")
    print(f"验证向量 {vec_path}  {vec_path.stat().st_size / 1024:.1f} KB", flush=True)


if __name__ == "__main__":
    main()
