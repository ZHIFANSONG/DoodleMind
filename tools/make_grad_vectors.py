"""生成「梯度参考向量」：用 PyTorch 自动微分算出 ∂logit/∂input，供 TS 侧反向传播比对。

为什么要这个：`src/lib/model.ts` 里手写了反向传播（四种算子的转置卷积 + maxpool
argmax 回路由 + ReLU 掩码），任何一处下标写反都不会崩，只会让 X-Ray 热力图指错
位置——"指错"在热力图上看起来永远都挺像那么回事，肉眼几乎抓不到。

有限差分能做校验，但在 ReLU 折点附近会被 kink 干扰（实测残差 4~20%），分辨不出
"实现有 bug" 还是 "差分本身不精确"。所以正道和前面三道闸门一样：让两个独立实现
（PyTorch autograd vs 手写 TS）算同一个数，比。

产物 `tools/vectors/grad_vectors.json` 里同时存 logits —— 校验脚本会先比 logits
再比梯度。logits 对不上说明这份 fixture 和端上模型不是同一个网络（op 序列漂了），
那就该先修 fixture，而不是怀疑梯度。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from export_model import (  # noqa: E402
    OUT,
    T_DENSE,
    T_DW,
    T_PW,
    DoodleNetSep,
    fold_bn,
    quantize_weight,
    reference_logits,
)
from raster import rasterize_u8  # noqa: E402

VEC_DIR = OUT.parent / "vectors"
SAMPLES = 4


def build_dequant_ops(model: DoodleNetSep) -> list[dict]:
    """复刻 export_model.main 的 op 展开 + 量化，得到端上真正在跑的那套权重。

    这段和 export_model.py 里的展开逻辑是**同一套规则的两处实现**：如果哪天
    那边加了新算子而这里没跟，fixture 里的 logits 就会和端上模型对不上，
    校验脚本第一件事就是比 logits —— 会当场炸，不会静默漂。
    """
    seq = [
        ("dense", model.b1),
        ("dw", model.b2), ("pw", model.b2),
        ("dw", model.b3), ("pw", model.b3),
        ("dw", model.b4), ("pw", model.b4),
    ]
    ops: list[dict] = []
    for kind, blk in seq:
        if kind == "dense":
            w, b = fold_bn(blk.conv, blk.bn)
            t, cin_, cout_ = T_DENSE, w.shape[1], w.shape[0]
        elif kind == "dw":
            w, b = fold_bn(blk.dw, blk.dw_bn)
            t, cin_, cout_ = T_DW, w.shape[0], w.shape[0]
        else:
            w, b = fold_bn(blk.pw, blk.pw_bn)
            t, cin_, cout_ = T_PW, w.shape[1], w.shape[0]
        ops.append({"type": t, "cin": cin_, "cout": cout_, "w": w, "b": b, "pool": False})

    for i in (0, 2, 4):
        ops[i]["pool"] = True

    decq = []
    for op in ops:
        qw, wscale = quantize_weight(op["w"])
        w_f32 = qw.astype(np.float32) * wscale.reshape((-1,) + (1,) * (qw.ndim - 1))
        decq.append({"type": op["type"], "cin": op["cin"], "cout": op["cout"], "w": w_f32, "b": op["b"], "pool": op["pool"]})
    return decq


def main() -> None:
    ck = torch.load(OUT / "doodlenet_sep.pt", map_location="cpu", weights_only=False)
    words = ck["words"]
    model = DoodleNetSep(len(words), width=tuple(ck["width"]))
    model.load_state_dict(ck["state"])
    model.eval()

    ops = build_dequant_ops(model)
    hw = model.head.weight.detach().numpy().astype(np.float32)
    hb = model.head.bias.detach().numpy().astype(np.float32)
    hq, hscale = quantize_weight(hw[:, None, :])
    hw_deq = hq.reshape(hw.shape).astype(np.float32) * hscale[:, None]

    vectors = json.loads((OUT.parent / "vectors" / "raster_vectors.json").read_text(encoding="utf-8"))
    picked = vectors["vectors"][:SAMPLES]

    out = []
    for i, item in enumerate(picked):
        img = rasterize_u8(item["strokes"])
        x = torch.tensor(np.asarray(img, dtype=np.float32).reshape(1, 1, 32, 32) / 255.0, requires_grad=True)
        logits = reference_logits(x, ops, hw_deq, hb)
        cls = int(logits[0].argmax())
        logits[0, cls].backward()
        grad = x.grad.detach().numpy().reshape(-1)
        out.append(
            {
                "word": item["word"],
                "cls": cls,
                "clsEn": words[cls],
                "logits": [round(float(v), 6) for v in logits[0].detach().numpy()],
                "grad": [round(float(v), 8) for v in grad],
                "strokes": item["strokes"],
            }
        )
        print(f"  样例 {i}: {item['word']:12s} cls={words[cls]:12s} |grad|max={np.abs(grad).max():.4f}", flush=True)

    VEC_DIR.mkdir(parents=True, exist_ok=True)
    path = VEC_DIR / "grad_vectors.json"
    path.write_text(json.dumps({"size": 32, "vectors": out}, ensure_ascii=False), encoding="utf-8")
    print(f"写出 {path} ({path.stat().st_size} 字节, {len(out)} 个样例)", flush=True)


if __name__ == "__main__":
    main()
