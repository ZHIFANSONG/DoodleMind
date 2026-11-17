/**
 * 手绘画板。
 *
 * 三个关键决定：
 *
 * 1. **画布是正方形**。模型输入是 32x32 正方形，QuickDraw 数据也全是正方形构图。
 *    画布做成非正方形的话，栅格化时按包围盒等比缩放会把内容压扁，模型的判断会
 *    悄悄变差，而且极难察觉。锁定 1:1 一劳永逸。
 *
 * 2. **点存归一化坐标 [0,1]**。这样换设备、换屏幕尺寸、存历史记录、重绘缩略图，
 *    全都不用换算。
 *
 * 3. **显示用 Catmull-Rom 平滑，推理用原始折线**。Catmull-Rom 是插值样条（曲线
 *    穿过原始点），所以显示出来的形状和存下来的几何是同一根线，做平滑只是让
 *    手指抖动的采样点看起来顺滑，不会改变模型看到的东西。
 *
 * 为什么用 View 的 responder 而不是 PanResponder.create：
 * PanResponder 实例通常要塞进 useMemo（否则每次渲染都重建 responder 配置），
 * 而 useMemo 的工厂函数在渲染期执行，一旦里面碰了 ref，React 的 lint 会（正确地）
 * 判定为「渲染期读 ref」。View 的 onResponder* props 就是普通回调属性，每次渲染
 * 传新的闭包即可，不需要任何 memo，也就不需要那套「最新值快照」的绕法。
 *
 * 仍然存在的那个 ref（activeRef）只在事件回调里读写——事件一定发生在渲染之后，
 * 这是允许的。它解决的是另一个问题：手指移动的事件频率高于 React 的重渲染频率，
 * 闭包里的 `active` state 会落后一两个点，用它累积笔画会丢点。
 */

import { memo, useCallback, useMemo, useRef, useState } from 'react';
import {
  StyleSheet,
  View,
  type GestureResponderEvent,
  type LayoutChangeEvent,
} from 'react-native';
import Svg, { Path } from 'react-native-svg';

export type Pt = { x: number; y: number };
export type Stroke = { pts: Pt[]; color: string; width: number };

type Props = {
  strokes: Stroke[];
  color: string;
  /** 归一化笔宽基准（相对画布边长的比例） */
  brushScale?: number;
  enabled?: boolean;
  onStrokeStart?: () => void;
  /** 一笔结束（抬手）时回调，用于触发推理 */
  onStrokeEnd?: (strokes: Stroke[]) => void;
  /** 正在画的时候每来一个点回调一次，用于实时推理节流 */
  onPoint?: (strokes: Stroke[]) => void;
  children?: React.ReactNode;
};

/** Catmull-Rom → 三次贝塞尔。穿过所有原始点。 */
export function toPathD(pts: Pt[], size: number): string {
  if (pts.length === 0) return '';
  const f = (v: number) => (v * size).toFixed(2);
  if (pts.length === 1) {
    // 单点：画一个极短线段，配合 round cap 变成一个点
    const x = pts[0].x * size;
    const y = pts[0].y * size;
    return `M${x.toFixed(2)},${y.toFixed(2)} L${(x + 0.01).toFixed(2)},${y.toFixed(2)}`;
  }
  if (pts.length === 2) {
    return `M${f(pts[0].x)},${f(pts[0].y)} L${f(pts[1].x)},${f(pts[1].y)}`;
  }

  let d = `M${f(pts[0].x)},${f(pts[0].y)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] ?? pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] ?? p2;

    // Catmull-Rom 转贝塞尔控制点（张力 1/6 对应标准 CR 样条）
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;

    d += ` C${f(c1x)},${f(c1y)} ${f(c2x)},${f(c2y)} ${f(p2.x)},${f(p2.y)}`;
  }
  return d;
}

const StrokePath = memo(function StrokePath({ stroke, size }: { stroke: Stroke; size: number }) {
  const d = useMemo(() => toPathD(stroke.pts, size), [stroke.pts, size]);
  return (
    <Path
      d={d}
      stroke={stroke.color}
      strokeWidth={stroke.width}
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  );
});

/** 手指静止时同一像素会涌出大量重复点：既浪费栅格化时间，又让 CR 样条出尖刺 */
const MIN_STEP_SQ = 0.00004;

export default function DrawingCanvas({
  strokes,
  color,
  brushScale = 0.028,
  enabled = true,
  onStrokeStart,
  onStrokeEnd,
  onPoint,
  children,
}: Props) {
  const [size, setSize] = useState(0);
  const [active, setActive] = useState<Stroke | null>(null);
  const activeRef = useRef<Stroke | null>(null);

  const onLayout = useCallback((e: LayoutChangeEvent) => {
    setSize(e.nativeEvent.layout.width);
  }, []);

  const pointOf = useCallback(
    (e: GestureResponderEvent): Pt => ({
      x: e.nativeEvent.locationX / size,
      y: e.nativeEvent.locationY / size,
    }),
    [size],
  );

  const begin = useCallback(
    (e: GestureResponderEvent) => {
      if (!enabled || size === 0) return;
      const st: Stroke = {
        pts: [pointOf(e)],
        color,
        width: Math.max(2, size * brushScale),
      };
      activeRef.current = st;
      setActive(st);
      onStrokeStart?.();
    },
    [enabled, size, pointOf, color, brushScale, onStrokeStart],
  );

  const extend = useCallback(
    (e: GestureResponderEvent) => {
      const cur = activeRef.current;
      if (!cur) return;
      const pt = pointOf(e);
      const last = cur.pts[cur.pts.length - 1];
      const dx = pt.x - last.x;
      const dy = pt.y - last.y;
      if (dx * dx + dy * dy < MIN_STEP_SQ) return;

      const next: Stroke = { ...cur, pts: [...cur.pts, pt] };
      activeRef.current = next;
      setActive(next);
      onPoint?.([...strokes, next]);
    },
    [pointOf, onPoint, strokes],
  );

  const end = useCallback(() => {
    const cur = activeRef.current;
    activeRef.current = null;
    setActive(null);
    if (!cur) return;
    onStrokeEnd?.([...strokes, cur]);
  }, [onStrokeEnd, strokes]);

  return (
    <View
      style={styles.wrap}
      onLayout={onLayout}
      onStartShouldSetResponder={() => enabled}
      onMoveShouldSetResponder={() => enabled}
      onResponderGrant={begin}
      onResponderMove={extend}
      onResponderRelease={end}
      onResponderTerminate={end}
      onResponderTerminationRequest={() => false}>
      {size > 0 && (
        <Svg width={size} height={size} style={StyleSheet.absoluteFill} pointerEvents="none">
          {strokes.map((s, i) => (
            <StrokePath key={i} stroke={s} size={size} />
          ))}
          {active && <StrokePath stroke={active} size={size} />}
        </Svg>
      )}
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1 },
});
