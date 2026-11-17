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
 * 关于 `latest` 这个 ref：PanResponder 的回调是在组件渲染之外被系统调用的，
 * 如果直接把 props 闭包进 useMemo 的 responder，就会永久持有第一次渲染时的
 * strokes/color/回调——画画时颜色改了却不出效果，就是这个原因。
 * 所以用一个「最新值」快照，在 effect 里同步（effect 在 commit 之后跑，
 * 而手势回调一定晚于 commit，时序上是安全的）。
 */

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PanResponder, StyleSheet, View, type LayoutChangeEvent } from 'react-native';
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

  /** 手势回调看到的最新世界状态；只在 effect / 事件里读写 */
  const latest = useRef({
    strokes,
    color,
    enabled,
    brushScale,
    size: 0,
    onPoint,
    onStrokeEnd,
    onStrokeStart,
  });
  const activeRef = useRef<Stroke | null>(null);

  useEffect(() => {
    latest.current = {
      ...latest.current,
      strokes,
      color,
      enabled,
      brushScale,
      onPoint,
      onStrokeEnd,
      onStrokeStart,
    };
  });

  const onLayout = useCallback((e: LayoutChangeEvent) => {
    const { width } = e.nativeEvent.layout;
    latest.current.size = width;
    setSize(width);
  }, []);

  const responder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => latest.current.enabled,
        onMoveShouldSetPanResponder: () => latest.current.enabled,
        onPanResponderGrant: (e) => {
          const ctx = latest.current;
          if (!ctx.enabled || ctx.size === 0) return;
          const pt: Pt = {
            x: e.nativeEvent.locationX / ctx.size,
            y: e.nativeEvent.locationY / ctx.size,
          };
          const st: Stroke = {
            pts: [pt],
            color: ctx.color,
            width: Math.max(2, ctx.size * ctx.brushScale),
          };
          activeRef.current = st;
          setActive(st);
          ctx.onStrokeStart?.();
        },
        onPanResponderMove: (e) => {
          const cur = activeRef.current;
          const ctx = latest.current;
          if (!cur || ctx.size === 0) return;
          const pt: Pt = {
            x: e.nativeEvent.locationX / ctx.size,
            y: e.nativeEvent.locationY / ctx.size,
          };
          // 距离过近的点丢掉：手指静止时会涌出大量重复点，
          // 既浪费栅格化时间，又让 Catmull-Rom 出现尖刺
          const last = cur.pts[cur.pts.length - 1];
          const dx = pt.x - last.x;
          const dy = pt.y - last.y;
          if (dx * dx + dy * dy < 0.00004) return;

          const next: Stroke = { ...cur, pts: [...cur.pts, pt] };
          activeRef.current = next;
          setActive(next);
          ctx.onPoint?.([...ctx.strokes, next]);
        },
        onPanResponderRelease: () => {
          const cur = activeRef.current;
          activeRef.current = null;
          setActive(null);
          if (!cur) return;
          const merged = [...latest.current.strokes, cur];
          latest.current.strokes = merged;
          latest.current.onStrokeEnd?.(merged);
        },
        onPanResponderTerminate: () => {
          const cur = activeRef.current;
          activeRef.current = null;
          setActive(null);
          if (!cur) return;
          const merged = [...latest.current.strokes, cur];
          latest.current.strokes = merged;
          latest.current.onStrokeEnd?.(merged);
        },
      }),
    [],
  );

  return (
    <View style={styles.wrap} onLayout={onLayout} {...responder.panHandlers}>
      {size > 0 && (
        <Svg width={size} height={size} style={StyleSheet.absoluteFill}>
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
