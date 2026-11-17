"""导出「栅格化一致性」测试向量。

为什么需要这个：`tools/check-engine.mjs` 喂给 TS 引擎的是**已经栅格化好的**
uint8 图像，也就是说它验证了推理，但**没有验证栅格化**。而栅格化是训练侧和运行时
唯一共享的几何口径——这里差一个偏移量，模型就会拿到和训练分布不同的输入，
表现为「模型突然变笨」而没有任何报错。

所以这里挑 40 张真实 QuickDraw 笔画，用 `tools/raster.py`（权威实现）算一遍，
把**原始笔画 + Python 的结果**一起写出去，让 TS 侧拿同样的原始笔画再算一遍比对。
"""

from __future__ import annotations

import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from raster import SIZE, rasterize_u8  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# 必须入库：CI 上要跑 check-raster.mjs，没有 Python 环境。
OUT = ROOT / "tools" / "vectors" / "raster_vectors.json"
N = 40


def main() -> None:
    # 从几个类别各拿一些，覆盖不同笔数/不同包围盒位置
    words = ["cat", "house", "bicycle", "apple", "umbrella", "star", "fish"]
    picked: list[dict] = []
    rng = random.Random(20260117)

    for word in words:
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{word}.ndjson"
        req = urllib.request.Request(url, headers={"Range": "bytes=0-400000"})
        text = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
        lines = [ln for ln in text.split("\n") if ln.strip().startswith("{")]
        parsed = []
        for ln in lines:
            try:
                parsed.append(json.loads(ln))
            except json.JSONDecodeError:
                continue  # 最后一行通常被截断
        rng.shuffle(parsed)
        for d in parsed[: max(2, N // len(words))]:
            drawing = d["drawing"]
            picked.append({"word": d["word"], "strokes": drawing})

    # 再加几张**人工构造的偏心样本**：QuickDraw 的数据基本都贴左上角（min=0），
    # 光靠它测不出「忘了减包围盒原点」这个 bug。用户的手指可能落在画布任何位置。
    picked.append({"word": "_offset", "strokes": [[[40, 40, 60, 60], [30, 60, 30, 60]]]})
    picked.append({"word": "_offset", "strokes": [[[100, 130, 160, 190], [80, 120, 80, 120]]]})
    picked.append({"word": "_tiny", "strokes": [[[200, 202, 205], [200, 201, 205]]]})

    vectors = []
    for p in picked:
        img = rasterize_u8(p["strokes"], SIZE)
        vectors.append(
            {
                "word": p["word"],
                "strokes": p["strokes"],
                "expected": img,
                "ink": sum(img),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"size": SIZE, "vectors": vectors}, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT}  ({len(vectors)} vectors, {OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
