/**
 * 把笔画栅格化成 32x32 灰度图。
 *
 * ⚠️ 本文件与 `tools/raster.py` 是**逐行等价**的两份实现：Python 版负责生成训练集，
 * TS 版负责运行时。任何一边改了参数或算法，另一边必须同步，否则训练分布和推理分布
 * 会产生 domain gap，模型会突然变笨而且很难查。
 *
 * 口径（与 QuickDraw simplified 一致）：
 *   1. 取所有笔画的点，算包围盒
 *   2. 等比缩放，让长边等于 SIZE - 2 * PAD
 *   3. 在正方形里居中
 *   4. 每个线段按「点到线段距离」画成半径 RADIUS 的圆头笔，覆盖率取 max 抗锯齿
 */

export const RASTER_SIZE = 32;
export const RASTER_PAD = 3;
export const STROKE_RADIUS = 1;

/** 一笔 = [xs[], ys[]]，与 QuickDraw 的 drawing 格式一致 */
export type Stroke = number[][];

function distToSegment(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const vx = px - ax;
  const vy = py - ay;
  const denom = dx * dx + dy * dy;
  if (denom <= 1e-12) return Math.sqrt(vx * vx + vy * vy);
  let t = (vx * dx + vy * dy) / denom;
  if (t < 0) t = 0;
  else if (t > 1) t = 1;
  const ex = vx - t * dx;
  const ey = vy - t * dy;
  return Math.sqrt(ex * ex + ey * ey);
}

/**
 * @param strokes 归一化到 [0,1] 或任意尺度的笔画（只关心相对几何，所以尺度无关）
 * @param out 长度 >= size*size 的输出缓冲，会被覆盖写
 * @returns 灰度值写入 out，范围 [0,1]，行主序
 */
export function rasterizeInto(
  strokes: Stroke[],
  out: Float32Array,
  size: number = RASTER_SIZE,
  pad: number = RASTER_PAD,
  radius: number = STROKE_RADIUS,
): void {
  out.fill(0, 0, size * size);

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let any = false;
  for (let s = 0; s < strokes.length; s++) {
    const st = strokes[s];
    if (st.length < 2) continue;
    const xs = st[0];
    const ys = st[1];
    for (let i = 0; i < xs.length; i++) {
      const x = xs[i];
      const y = ys[i];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      any = true;
    }
  }
  if (!any) return;

  const w = maxX - minX;
  const h = maxY - minY;
  const span = Math.max(w, h, 1e-6);
  const scale = (size - 2 * pad) / span;
  const offX = pad + ((size - 2 * pad) - w * scale) / 2;
  const offY = pad + ((size - 2 * pad) - h * scale) / 2;

  for (let s = 0; s < strokes.length; s++) {
    const st = strokes[s];
    if (st.length < 2) continue;
    const xs = st[0];
    const ys = st[1];
    const n = xs.length;
    if (n === 0) continue;

    // 单点笔画退化成一个零长线段，保证能画出圆点
    const count = n === 1 ? 1 : n - 1;
    for (let i = 0; i < count; i++) {
      const ax = xs[i] * scale + offX;
      const ay = ys[i] * scale + offY;
      const bx = n === 1 ? ax : xs[i + 1] * scale + offX;
      const by = n === 1 ? ay : ys[i + 1] * scale + offY;

      const loX = Math.max(0, Math.floor(Math.min(ax, bx) - radius - 1));
      const hiX = Math.min(size - 1, Math.floor(Math.max(ax, bx) + radius + 1));
      const loY = Math.max(0, Math.floor(Math.min(ay, by) - radius - 1));
      const hiY = Math.min(size - 1, Math.floor(Math.max(ay, by) + radius + 1));

      for (let py = loY; py <= hiY; py++) {
        const cy = py + 0.5;
        const rowBase = py * size;
        for (let px = loX; px <= hiX; px++) {
          const cx = px + 0.5;
          const d = distToSegment(cx, cy, ax, ay, bx, by);
          let cov = radius + 0.5 - d;
          if (cov <= 0) continue;
          if (cov > 1) cov = 1;
          const idx = rowBase + px;
          if (cov > out[idx]) out[idx] = cov;
        }
      }
    }
  }
}

/** 便捷版：返回 0..255 的 Uint8Array（网络输入的原始量纲） */
export function rasterize(
  strokes: Stroke[],
  size: number = RASTER_SIZE,
): Uint8Array {
  const f = new Float32Array(size * size);
  rasterizeInto(strokes, f, size);
  const u8 = new Uint8Array(size * size);
  for (let i = 0; i < f.length; i++) {
    const v = Math.round(f[i] * 255);
    u8[i] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  return u8;
}

/** 墨迹总像素量，用来判断「基本没画东西」 */
export function inkAmount(img: Uint8Array): number {
  let s = 0;
  for (let i = 0; i < img.length; i++) s += img[i];
  return s / 255;
}
