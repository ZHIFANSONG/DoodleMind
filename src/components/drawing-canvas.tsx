/**
 * 手绘画板。
 *
 * 四个关键决定：
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
 * 4. **笔宽随速度变化**（见下方 strokeSegments）。这是「像笔」和「像折线」的分界。
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

/** `t` 是采样时刻（毫秒），只有实时笔迹有；历史记录里不存，重绘时退化为等宽 */
export type Pt = { x: number; y: number; t?: number };
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

/* ────────────────────────── 几何 ────────────────────────── */

/**
 * Catmull-Rom 控制点。返回每段的三次贝塞尔 c1/c2。
 *
 * 标准 CR 样条的切向量是 (p[i+1] - p[i-1]) / 2，转成贝塞尔控制点要再乘 1/3，
 * 合起来就是 /6。端点用外插（p0 = p1）而不是镜像，避免起笔处出现甩尾。
 */
function crControls(pts: Pt[], i: number): { c1x: number; c1y: number; c2x: number; c2y: number } {
  const p0 = pts[i - 1] ?? pts[i];
  const p1 = pts[i];
  const p2 = pts[i + 1];
  const p3 = pts[i + 2] ?? p2;
  return {
    c1x: p1.x + (p2.x - p0.x) / 6,
    c1y: p1.y + (p2.y - p0.y) / 6,
    c2x: p2.x - (p3.x - p1.x) / 6,
    c2y: p2.y - (p3.y - p1.y) / 6,
  };
}

/** 整条笔迹一条 path。给画册缩略图用（那里只需要形状，不需要压感）。 */
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
    const { c1x, c1y, c2x, c2y } = crControls(pts, i);
    d += ` C${f(c1x)},${f(c1y)} ${f(c2x)},${f(c2y)} ${f(pts[i + 1].x)},${f(pts[i + 1].y)}`;
  }
  return d;
}

/**
 * 按速度给每一段分配笔宽，再把相邻同宽的段合并成一条 path。
 *
 * **`stroke.width` 是比例，不是像素。** 这一点踩过坑：早先它存的是
 * 「创建这一笔时画布的像素宽度」（约 14px），画册拿同一份数据去画 370px 的缩略图时
 * 又乘了一次尺寸，笔画变成 4600px 粗，整张缩略图糊成一块黑方块——而且没有任何报错。
 * 存比例（相对画布边长）才能让同一条笔迹在任何尺寸下都长对样子，
 * 这正是「点存归一化坐标」的同一个理由。
 *
 * 为什么不逐段 `<Path>`：一条 200 点的笔画就是 200 个 React 元素，手指移动时
 * 每帧重建，必掉帧。为什么不画成多边形轮廓：那才是"真正的墨"，但每次移动都要
 * 重算法线 + 上下沿，字符串长度翻倍，在 JS 里算不动。
 *
 * 折中方案是**分档**：把宽度量化成 4 档，相邻同档合并。一条笔画最终只有 2–5 个
 * Path，视觉上已经是"起笔粗、走笔细、收笔再粗"的毛笔感，成本几乎为零。
 *
 * 速度映射：慢 → 墨堆得多 → 粗；快 → 飞白 → 细。
 * 用绝对速度（px/ms）而不是"本笔中位速度"，因为中位数会随笔画变长而漂移，
 * 导致先画的段每帧都在改宽度，看起来像在呼吸。
 */
const WIDTH_LEVELS = 4;
const SPEED_FULL = 1.4; // px/ms 之上算全速
const THICK_RATIO = 1.22; // 最慢时的笔宽倍数
const THIN_RATIO = 0.58; // 最快时的笔宽倍数
/** 宽度 EMA 平滑系数：手指速度抖动很剧烈，不平滑就是锯齿状粗细 */
const WIDTH_EMA = 0.45;

export type InkSegment = { d: string; w: number };

export function strokeSegments(stroke: Stroke, size: number): InkSegment[] {
  const { pts, width } = stroke;
  if (pts.length === 0) return [];
  const f = (v: number) => (v * size).toFixed(2);
  // width 是比例 → 先换成这个尺寸下的像素。下限 1.2px 保证极细的画布上仍可见。
  const basePx = Math.max(1.2, width * size);

  if (pts.length === 1) {
    const x = pts[0].x * size;
    const y = pts[0].y * size;
    return [{ d: `M${x.toFixed(2)},${y.toFixed(2)} L${(x + 0.01).toFixed(2)},${y.toFixed(2)}`, w: basePx }];
  }

  // 逐点速度（px/ms）。历史记录没有时间戳，退化成匀速 → 单一档位。
  const hasTime = pts[0].t !== undefined && pts[pts.length - 1].t !== undefined;
  const levels: number[] = []; // 每段一个档位 0..WIDTH_LEVELS-1
  let ema = 0.5;

  for (let i = 0; i < pts.length - 1; i++) {
    let speed = 0;
    if (hasTime) {
      const dt = Math.max(1, (pts[i + 1].t ?? 0) - (pts[i].t ?? 0));
      const dx = (pts[i + 1].x - pts[i].x) * size;
      const dy = (pts[i + 1].y - pts[i].y) * size;
      speed = Math.sqrt(dx * dx + dy * dy) / dt;
    }
    // 0 = 最慢（粗），1 = 最快（细）
    const norm = Math.min(1, speed / SPEED_FULL);
    ema += (norm - ema) * WIDTH_EMA;
    // 档位：0..3，越大越细。用 (1-ema) 让慢速落在高档位（粗）。
    const lv = Math.min(WIDTH_LEVELS - 1, Math.floor((1 - ema) * WIDTH_LEVELS));
    levels.push(lv);
  }

  // 合并相邻同档
  const out: InkSegment[] = [];
  let i = 0;
  while (i < levels.length) {
    let j = i;
    while (j + 1 < levels.length && levels[j + 1] === levels[i]) j++;

    let d = `M${f(pts[i].x)},${f(pts[i].y)}`;
    for (let k = i; k <= j; k++) {
      const { c1x, c1y, c2x, c2y } = crControls(pts, k);
      d += ` C${f(c1x)},${f(c1y)} ${f(c2x)},${f(c2y)} ${f(pts[k + 1].x)},${f(pts[k + 1].y)}`;
    }
    // lv 越大越细：lv=0 → THICK_RATIO，lv=3 → THIN_RATIO
    const ratio =
      THICK_RATIO + (THIN_RATIO - THICK_RATIO) * (levels[i] / (WIDTH_LEVELS - 1));
    out.push({ d, w: basePx * ratio });
    i = j + 1;
  }
  return out;
}

const StrokeInk = memo(function StrokeInk({ stroke, size }: { stroke: Stroke; size: number }) {
  const segs = useMemo(() => strokeSegments(stroke, size), [stroke, size]);
  return (
    <>
      {segs.map((s, i) => (
        <Path
          key={i}
          d={s.d}
          stroke={stroke.color}
          strokeWidth={s.w}
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ))}
    </>
  );
});

/* ────────────────────────── 组件 ────────────────────────── */

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
      // 触摸事件的 timestamp 是设备开机以来的毫秒数，同一次手势内自洽，
      // 正好用来算瞬时速度；退回 Date.now() 也不影响相对差值。
      t: e.nativeEvent.timestamp ?? Date.now(),
    }),
    [size],
  );

  const begin = useCallback(
    (e: GestureResponderEvent) => {
      if (!enabled || size === 0) return;
      const st: Stroke = {
        pts: [pointOf(e)],
        color,
        // 存比例（相对画布边长），不存像素——见 strokeSegments 顶部那段注释
        width: brushScale,
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
            <StrokeInk key={i} stroke={s} size={size} />
          ))}
          {active && <StrokeInk stroke={active} size={size} />}
        </Svg>
      )}
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1 },
});
