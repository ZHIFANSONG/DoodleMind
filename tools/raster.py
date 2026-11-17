"""把 QuickDraw 原始笔画栅格化成网络输入。

这个文件是**唯一权威实现**：`src/lib/raster.ts` 是它逐行等价的 TypeScript 移植，
两边任何改动都必须同步，否则训练/推理会串味（domain gap）。
"""

from __future__ import annotations

from typing import Iterable, Sequence

# 输入尺寸（正方形边长，像素）
SIZE = 32
# 画布内边距：图形最大边占据 SIZE - 2 * PAD
PAD = 3.0
# 线的半径（像素），控制笔画粗细
STROKE_RADIUS = 1.0


def _dist_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    vx = px - ax
    vy = py - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return (vx * vx + vy * vy) ** 0.5
    t = (vx * dx + vy * dy) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    ex = vx - t * dx
    ey = vy - t * dy
    return (ex * ex + ey * ey) ** 0.5


def rasterize(
    strokes: Sequence[Sequence[Sequence[float]]],
    size: int = SIZE,
    pad: float = PAD,
    radius: float = STROKE_RADIUS,
) -> list[float]:
    """返回 size*size 的灰度浮点数组，取值 [0, 1]，按行优先（row-major）。"""
    canvas = [0.0] * (size * size)

    xs = []
    ys = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        xs.extend(stroke[0])
        ys.extend(stroke[1])
    if not xs:
        return canvas

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x
    h = max_y - min_y
    span = max(w, h, 1e-6)
    scale = (size - 2.0 * pad) / span
    # 居中：先平移到包围盒原点，再缩放到 (size-2pad) 的方框里居中
    off_x = pad + ((size - 2.0 * pad) - w * scale) / 2.0
    off_y = pad + ((size - 2.0 * pad) - h * scale) / 2.0

    def project(stroke):
        sx = stroke[0]
        sy = stroke[1]
        return [
            ((sx[i] - min_x) * scale + off_x, (sy[i] - min_y) * scale + off_y)
            for i in range(len(sx))
        ]

    for stroke in strokes:
        pts = project(stroke)
        if len(pts) == 1:
            pts = [pts[0], pts[0]]
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            lo_x = max(0, int(min(ax, bx) - radius - 1.0))
            hi_x = min(size - 1, int(max(ax, bx) + radius + 1.0))
            lo_y = max(0, int(min(ay, by) - radius - 1.0))
            hi_y = min(size - 1, int(max(ay, by) + radius + 1.0))
            for py in range(lo_y, hi_y + 1):
                cy = py + 0.5
                for px in range(lo_x, hi_x + 1):
                    cx = px + 0.5
                    d = _dist_to_segment(cx, cy, ax, ay, bx, by)
                    cov = radius + 0.5 - d
                    if cov <= 0.0:
                        continue
                    if cov > 1.0:
                        cov = 1.0
                    idx = py * size + px
                    if cov > canvas[idx]:
                        canvas[idx] = cov
    return canvas


def rasterize_u8(strokes, size: int = SIZE) -> list[int]:
    """栅格化成 0..255 的整数数组（训练数据与 TS 运行时都用这个量纲）。"""
    return [min(255, int(round(v * 255.0))) for v in rasterize(strokes, size)]


def bbox(strokes) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for stroke in strokes:
        xs.extend(stroke[0])
        ys.extend(stroke[1])
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def ascii_preview(img: Sequence[int], size: int = SIZE) -> str:
    ramp = " .:-=+*#%@"
    out = []
    for y in range(size):
        row = "".join(
            ramp[min(len(ramp) - 1, int(img[y * size + x]) * len(ramp) // 256)] for x in range(size)
        )
        out.append(row)
    return "\n".join(out)
