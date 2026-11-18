/**
 * X-Ray：把「模型在看哪儿」画在画上。
 *
 * 底图是 32×32 的 ∂logit/∂像素，但它不能直接铺上去——直接铺出来是一堆 25px 的
 * 马赛克方块，而且峰值永远只有一个像素最亮，看不出"区域"。所以做了三件事：
 *
 * 1. **降到 16×16 取块内最大值**（不是平均值）。平均会把一整个亮块稀释掉，而
 *    我想表达的是"这块区域里有它在意的笔画"，取最大值才对得上。
 * 2. **透明度用平方根压动态范围**。梯度分布极偏（一个尖峰 + 大片接近零），
 *    线性映射的结果只有几个格子看得见。
 * 3. **只画到 55% 不透明度**。热力图是叠加层，它不该把用户自己画的墨盖掉——
 *    这张图的主角仍然是那笔画。
 */
import { memo } from 'react';
import Svg, { Rect } from 'react-native-svg';
import { StyleSheet } from 'react-native';

const SRC = 32;
const CELLS = 16;
const STEP = SRC / CELLS;

type Props = {
  /** 32×32 的显著性，0..1；null 表示这次不显示 */
  map: Float32Array | null;
  /** 画布边长（屏幕像素） */
  size: number;
  /** 叠加强度 0..1 */
  intensity?: number;
};

export const SaliencyOverlay = memo(function SaliencyOverlay({ map, size, intensity = 0.55 }: Props) {
  if (!map || size <= 0) return null;
  const cell = size / CELLS;
  const rects = [];
  for (let cy = 0; cy < CELLS; cy++) {
    for (let cx = 0; cx < CELLS; cx++) {
      let m = 0;
      for (let sy = 0; sy < STEP; sy++) {
        const row = (cy * STEP + sy) * SRC + cx * STEP;
        for (let sx = 0; sx < STEP; sx++) {
          const v = map[row + sx];
          if (v > m) m = v;
        }
      }
      const a = m * m * intensity;
      if (a < 0.04) continue;
      rects.push(
        <Rect key={`${cy}-${cx}`} x={cx * cell} y={cy * cell} width={cell} height={cell} fill="#D9422B" opacity={a} />,
      );
    }
  }
  if (rects.length === 0) return null;
  return (
    <Svg width={size} height={size} style={styles.root} pointerEvents="none">
      {rects}
    </Svg>
  );
});

const styles = StyleSheet.create({
  root: { position: 'absolute', top: 0, left: 0 },
});
