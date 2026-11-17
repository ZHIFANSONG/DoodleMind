"""
生成 App 图标 / 启动图 / favicon / Android 自适应图标。

图标里的那只猫**是用本项目的笔刷规则画的**：沿折线撒圆头笔、宽度随「速度」
变化（这里用人为设定的曲线代替手指速度），再用 4 倍超采样做抗锯齿。
和 `src/components/drawing-canvas.tsx::strokeSegments` 是同一个视觉逻辑——
图标本身就是一张出自这个 App 的涂鸦。

用法：python tools/make_icons.py
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, "assets", "images")

PAPER = (251, 250, 246)
INK = (23, 22, 26)
SEAL = (217, 66, 43)

SS = 4  # 超采样倍数


def cat_strokes() -> list[tuple[list[tuple[float, float]], float, float]]:
    """
    返回 [(点的列表(归一化0..1), 起始半宽(占边长比例), 结束半宽)]。

    构成：头一圈、两只耳、两点眼、一个鼻嘴、四根胡须。
    头故意画成微微不圆的椭圆并让首尾差一点点没接上——完全规整的话
    看起来像图标库里的几何图形，不像手指画的。
    """
    out: list[tuple[list[tuple[float, float]], float, float]] = []

    # 头：极坐标椭圆，半径带低频起伏。闭合画满一圈——
    # 留缺口在图标尺寸上看不出「手绘感」，只会像笔没接上。
    head: list[tuple[float, float]] = []
    n = 160
    for i in range(n + 1):
        a = 2 * math.pi * (i / n)
        wob = 1.0 + 0.013 * math.sin(3 * a + 0.7) + 0.007 * math.sin(5 * a + 2.1)
        rx = 0.238 * wob
        ry = 0.203 * wob
        head.append((0.5 + rx * math.cos(a), 0.548 + ry * math.sin(a)))
    out.append((head, 0.0175, 0.0175))

    # 左耳：起笔点要**正好落在头部轮廓线**上（x=0.388 处轮廓的 y≈0.369），
    # 这样圆头笔尖与轮廓重合、看不出接头。落在轮廓内侧会往外鼓一个小圆点，
    # 落在轮廓外侧会露出一截断头——都和右耳不对称。
    out.append(
        (
            [
                (0.388, 0.369),
                (0.340, 0.288),
                (0.318, 0.228),
                (0.352, 0.246),
                (0.418, 0.306),
                (0.448, 0.348),
            ],
            0.017,
            0.015,
        )
    )
    # 右耳
    out.append(
        (
            [
                (0.562, 0.352),
                (0.604, 0.302),
                (0.672, 0.244),
                (0.696, 0.232),
                (0.684, 0.292),
                (0.660, 0.396),
            ],
            0.017,
            0.015,
        )
    )

    # 眼睛：几乎不动的两个小点（真实点一下就是这种短促笔画）
    for ex in (0.412, 0.590):
        out.append(([(ex, 0.518), (ex, 0.538)], 0.018, 0.018))

    # 鼻：一个倒三角，单独一笔
    out.append(([(0.489, 0.592), (0.517, 0.592), (0.503, 0.612), (0.489, 0.592)], 0.009, 0.009))
    # 嘴：从鼻尖往下一竖，然后左右各一撇（猫的 ω 形）
    out.append(([(0.503, 0.612), (0.503, 0.630)], 0.0085, 0.0085))
    out.append(([(0.503, 0.630), (0.478, 0.646), (0.462, 0.630)], 0.0085, 0.006))
    out.append(([(0.503, 0.630), (0.528, 0.646), (0.544, 0.630)], 0.0085, 0.006))

    # 胡须：左右各两根，往外越来越细
    out.append(([(0.262, 0.572), (0.148, 0.545)], 0.0072, 0.0035))
    out.append(([(0.264, 0.622), (0.156, 0.642)], 0.0072, 0.0035))
    out.append(([(0.738, 0.568), (0.852, 0.540)], 0.0072, 0.0035))
    out.append(([(0.736, 0.618), (0.846, 0.638)], 0.0072, 0.0035))
    return out


def resample(pts: list[tuple[float, float]], step: float = 0.004) -> list[tuple[float, float]]:
    """按固定弧长重采样，让笔宽沿长度均匀变化（否则长段会突然变细）。"""
    if len(pts) < 2:
        return pts
    acc = [(pts[0][0], pts[0][1])]
    carry = 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-9:
            continue
        t = (step - carry) / seg
        while t <= 1.0:
            acc.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
            t += step / seg
        carry = (carry + seg) % step
    acc.append(pts[-1])
    return acc


def stamp(
    draw: ImageDraw.ImageDraw,
    pts: list[tuple[float, float]],
    size: int,
    w0: float,
    w1: float,
    fill: tuple[int, int, int, int],
    alpha_profile: bool = False,
) -> None:
    """
    沿折线撒圆头笔，半径从 w0 线性过渡到 w1。

    alpha_profile 默认关掉：起笔/收笔淡出在超采样缩放后会在端头留下一圈
    灰边（半透明墨叠在浅纸上就是灰），看起来像笔尖沾了脏东西。
    真实的压感变宽不需要这一层——宽度变化本身就已经表达了起收笔。
    """
    dense = resample(pts)
    total = len(dense) - 1
    if total <= 0:
        return
    for i, (x, y) in enumerate(dense):
        t = i / total
        r = (w0 + (w1 - w0) * t) * size
        a = 1.0
        if alpha_profile:
            edge = 0.12
            if t < edge:
                a = 0.55 + 0.45 * (t / edge)
            elif t > 1 - edge:
                a = 0.55 + 0.45 * ((1 - t) / edge)
        c = (fill[0], fill[1], fill[2], int(fill[3] * a))
        draw.ellipse((x * size - r, y * size - r, x * size + r, y * size + r), fill=c)


def draw_cat(size: int, color: tuple[int, int, int, int], scale: float = 1.0) -> Image.Image:
    """在白/透明底上画那只猫。返回 size×size 的 RGBA。"""
    big = size * SS
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for pts, w0, w1 in cat_strokes():
        # scale 让整只猫在画布里占的比重大小可调
        sp = [(0.5 + (x - 0.5) * scale, 0.5 + (y - 0.5) * scale + 0.02 * (1 - scale)) for x, y in pts]
        stamp(d, sp, big, w0 * scale, w1 * scale, color)
    return img.resize((size, size), Image.LANCZOS)


def solid(size: int, rgb: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGBA", (size, size), (*rgb, 255))


def write(img: Image.Image, name: str) -> None:
    path = os.path.join(IMAGES, name)
    img.save(path, "PNG", optimize=True)
    print(f"  {name}  {img.size[0]}x{img.size[1]}  {os.path.getsize(path) // 1024}KB")


def main() -> None:
    os.makedirs(IMAGES, exist_ok=True)

    # ── iOS / 通用图标：纸底 + 墨线，不留圆角（系统自己切）。
    # 不放任何彩色装饰点：右下角一个小圆点在图标尺寸下会被读成未读角标。
    icon = solid(1024, PAPER)
    icon.alpha_composite(draw_cat(1024, (*INK, 255), scale=1.0))
    write(icon.convert("RGB").convert("RGBA"), "icon.png")

    # ── 启动图：只有墨线，透明底（背景色由 app.json 的 splash 配置给）
    write(draw_cat(512, (*INK, 255), scale=1.0), "splash-icon.png")

    # ── favicon（web 预览用）
    write(draw_cat(192, (*INK, 255), scale=1.0), "favicon.png")

    # ── Android 自适应图标：前景必须留在中间 66% 安全区里
    write(draw_cat(512, (*INK, 255), scale=0.66), "android-icon-foreground.png")
    write(solid(512, PAPER), "android-icon-background.png")
    write(draw_cat(432, (0, 0, 0, 255), scale=0.66), "android-icon-monochrome.png")

    print("图标生成完毕")


if __name__ == "__main__":
    main()
