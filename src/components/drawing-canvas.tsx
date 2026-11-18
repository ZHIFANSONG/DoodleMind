/**
 * 手绘画板。
 *
 * 五个关键决定：
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
 * 5. **正在画的那一笔走「廉价通道」，抬手的瞬间才升级成「精细通道」。**
 *
 * 第 5 条是这一版重写的全部理由。之前每个触点都做三件昂贵的事：
 *   - `{...cur, pts: [...cur.pts, pt]}` —— 每来一个点就复制整条笔画，200 点的
 *     笔画画下来是 200 次、共 2 万个元素的内存搬运，复杂度 O(n²)；
 *   - `setActive(next)` → 重渲染 → `strokeSegments()` 把整条笔画重新分档、重新拼
 *     SVG 路径字符串（几十次 `toFixed`）——每一帧都做一遍全量；
 *   - `onPoint?.([...strokes, next])` —— 又一次全量复制，而推理本身还会同步占用
 *     JS 线程 8~20ms。
 *
 * 在 ProMotion（iPhone 16 Pro Max 是 120Hz）上，这些加起来就是「手指在动、
 * 墨跟不上」。现在：
 *   - 点推进一个**可变的**数组，路径字符串**只追加**，每点 O(1)；
 *   - 屏幕更新走 `requestAnimationFrame`，一帧最多一次，且只有一条 Path；
 *   - 递给推理的数组按时间节流（`EMIT_MS`），并在 `setTimeout(0)` 里递出去，
 *     让触摸事件先处理完，推理不阻塞落笔那一帧；
 *   - 抬手时才生成不可变的 `Stroke`，那时才算压感分档——算一次就够。
 *
 * 为什么用 View 的 responder 而不是 PanResponder.create：
 * PanResponder 实例通常要塞进 useMemo（否则每次渲染都重建 responder 配置），
 * 而 useMemo 的工厂函数在渲染期执行，一旦里面碰了 ref，React 的 lint 会（正确地）
 * 判定为「渲染期读 ref」。View 的 onResponder* props 就是普通回调属性，每次渲染
 * 传新的闭包即可，不需要任何 memo，也就不需要那套「最新值快照」的绕法。
 */

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
  /** 正在画的时候每隔 EMIT_MS 回调一次，用于实时推理 */
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
 * 为什么不逐段 `<Path>`：一条 200 点的笔画就是 200 个 React 元素。为什么不画成
 * 多边形轮廓：那才是"真正的墨"，但每次移动都要重算法线 + 上下沿，字符串长度翻倍。
 *
 * 折中方案是**分档**：把宽度量化成 4 档，相邻同档合并。一条笔画最终只有 2–5 个
 * Path，视觉上已经是"起笔粗、走笔细、收笔再粗"的毛笔感。
 *
 * 速度映射：慢 → 墨堆得多 → 粗；快 → 飞白 → 细。
 * 用绝对速度（px/ms）而不是"本笔中位速度"，因为中位数会随笔画变长而漂移，
 * 导致先画的段每帧都在改宽度，看起来像在呼吸。
 *
 * 这个函数只在**抬手时**跑一次（见文件头第 5 条）。
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
    return [
      { d: `M${x.toFixed(2)},${y.toFixed(2)} L${(x + 0.01).toFixed(2)},${y.toFixed(2)}`, w: basePx },
    ];
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
    const ratio = THICK_RATIO + (THIN_RATIO - THICK_RATIO) * (levels[i] / (WIDTH_LEVELS - 1));
    out.push({ d, w: basePx * ratio });
    i = j + 1;
  }
  return out;
}

/** 已抬手的笔画。只在这条笔画的引用变了时才重算路径（memo 兜住）。 */
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

/**
 * 递给推理的最小间隔。推理本身是同步的（设备上 15~30ms，Node 里 8ms），
 * 一秒最多来九次，剩下的时间全留给手指。为什么不再密一点：候选栏一秒刷
 * 九次，人眼已经分不出差别，但 JS 线程被同步推理占住的时长差别很明显。
 */
const EMIT_MS = 110;

/**
 * 屏幕上墨迹的刷新上限（毫秒）。ProMotion 是 120Hz，一帧 8ms；跟着 120Hz
 * 重渲染收益很小，成本翻倍，所以压到 60Hz 一带。
 */
const FRAME_MS = 15;

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
  /** 正在画的那一笔的 path 字符串。每帧最多更新一次。 */
  const [liveD, setLiveD] = useState('');

  /**
   * 可变笔画。**故意不进 state**：每来一个点就 setState 会让整页重渲染，
   * 而这条数据只在抬手时才有意义。
   */
  const liveRef = useRef<{ pts: Pt[]; d: string; lastEmit: number } | null>(null);
  const frameRef = useRef<number | null>(null);
  const lastFrameAt = useRef(0);

  /**
   * 回调与配置的最新值。触摸事件频率高于重渲染，闭包会落后；但事件一定发生在
   * 渲染之后，所以回调里读这个快照是安全的。用 effect 同步（渲染期写 ref 会被
   * React 的 lint 正确拦下）。
   */
  const latest = useRef({
    strokes,
    color,
    brushScale,
    enabled,
    onPoint,
    onStrokeEnd,
    onStrokeStart,
    size: 0,
  });
  useEffect(() => {
    latest.current = {
      ...latest.current,
      strokes,
      color,
      brushScale,
      enabled,
      onPoint,
      onStrokeEnd,
      onStrokeStart,
    };
  });

  useEffect(
    () => () => {
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    },
    [],
  );

  const scheduleFrame = useCallback(() => {
    if (frameRef.current !== null) return;
    const pump = () => {
      const now = Date.now();
      if (now - lastFrameAt.current < FRAME_MS) {
        // 这一帧离上次刷新太近，跳过；下一帧再来
        frameRef.current = requestAnimationFrame(pump);
        return;
      }
      frameRef.current = null;
      lastFrameAt.current = now;
      const cur = liveRef.current;
      // 一帧只把当前的完整 path 交给 React；中间那些点全部合并掉了
      setLiveD(cur ? cur.d : '');
    };
    frameRef.current = requestAnimationFrame(pump);
  }, []);

  const onLayout = useCallback((e: LayoutChangeEvent) => {
    const w = e.nativeEvent.layout.width;
    latest.current.size = w;
    setSize(w);
  }, []);

  const pointOf = useCallback((e: GestureResponderEvent): Pt => {
    const s = latest.current.size || 1;
    /*
     * 必须夹到 [0,1]。
     *
     * 手指按住画布后一路划出画布边界时，事件仍然会送到画布（响应者已经确定），
     * locationX 会大于 size 或小于 0。这些点如果原样记下来，栅格化时包围盒会被
     * 撑大、整幅画被压扁到一角——屏幕上又因为 overflow:hidden 看不到越界的墨，
     * 于是表现成「画着画着模型突然认不出来了」。夹住就等于墨停在边框上，
     * 和用户看到的一致。
     */
    const clamp = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
    return {
      x: clamp(e.nativeEvent.locationX / s),
      y: clamp(e.nativeEvent.locationY / s),
      // 触摸事件的 timestamp 是设备开机以来的毫秒数，同一次手势内自洽，
      // 正好用来算瞬时速度；退回 Date.now() 也不影响相对差值。
      t: e.nativeEvent.timestamp ?? Date.now(),
    };
  }, []);

  const begin = useCallback(
    (e: GestureResponderEvent) => {
      const { enabled: on, size: s } = latest.current;
      if (!on || !s) return;
      const pt = pointOf(e);
      liveRef.current = {
        pts: [pt],
        // 单点也要看得见：极短线段 + round cap = 一个点
        d: `M${(pt.x * s).toFixed(1)},${(pt.y * s).toFixed(1)} L${(pt.x * s + 0.1).toFixed(1)},${(
          pt.y * s
        ).toFixed(1)}`,
        lastEmit: 0,
      };
      setLiveD(liveRef.current.d);
      latest.current.onStrokeStart?.();
    },
    [pointOf],
  );

  const extend = useCallback(
    (e: GestureResponderEvent) => {
      const cur = liveRef.current;
      if (!cur) return;
      const s = latest.current.size || 1;
      const pt = pointOf(e);
      const last = cur.pts[cur.pts.length - 1];
      const dx = pt.x - last.x;
      const dy = pt.y - last.y;
      if (dx * dx + dy * dy < MIN_STEP_SQ) return;

      // O(1)：推进可变数组 + 追加路径字符串
      cur.pts.push(pt);
      cur.d += ` L${(pt.x * s).toFixed(1)},${(pt.y * s).toFixed(1)}`;
      scheduleFrame();

      const now = Date.now();
      if (now - cur.lastEmit < EMIT_MS) return;
      cur.lastEmit = now;
      const snapshot: Stroke = {
        pts: cur.pts.slice(),
        color: latest.current.color,
        width: latest.current.brushScale,
      };
      const all = [...latest.current.strokes, snapshot];
      // 挪出触摸回调：推理是同步的，直接在这里调会让落笔那一帧掉帧
      setTimeout(() => latest.current.onPoint?.(all), 0);
    },
    [pointOf, scheduleFrame],
  );

  const end = useCallback(() => {
    const cur = liveRef.current;
    liveRef.current = null;
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    setLiveD('');
    if (!cur) return;
    const stroke: Stroke = {
      pts: cur.pts,
      color: latest.current.color,
      // 存比例（相对画布边长），不存像素——见 strokeSegments 顶部那段注释
      width: latest.current.brushScale,
    };
    latest.current.onStrokeEnd?.([...latest.current.strokes, stroke]);
  }, []);

  // 用 props 里的 brushScale 算，不读 ref：渲染期读 ref 是被正确禁止的，
  // 而这里要的本来就是「这一次渲染的 brushScale」，不是最新快照。
  const liveWidth = Math.max(1.2, brushScale * size);

  /*
   * `onMoveShouldSetResponder` 返回 false 是必须的。
   *
   * 触摸可能从别的控件上开始、然后滑进画布（从判读区往下拖、从候选栏往上拖）。
   * 这时系统会来问画布「要不要接管这次手势」，若回答 true，画布会在**手指进入
   * 的那一刻**抢走它并从那里开始一条笔迹——表现就是「我明明在别处点了一下，
   * 画布自己画了一道」，起点还和手指对不上。画布只该回应从它自己身上开始的那一次。
   */
  return (
    <View
      style={styles.wrap}
      onLayout={onLayout}
      onStartShouldSetResponder={() => enabled}
      onMoveShouldSetResponder={() => false}
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
          {/*
            正在画的这一笔：一条 polyline，等宽。为什么不在这里也做 CR 平滑和压感
            分档——120Hz 采样下相邻点只差两三个像素，polyline 看起来已经是顺的；
            而分档要重算整条笔画，这正是卡顿的来源。抬手后立刻换成精细版，
            中间不会有一帧的空档：setLiveD('') 与 setStrokes(...) 在同一个批次里。
          */}
          {liveD !== '' && (
            <Path
              d={liveD}
              stroke={color}
              strokeWidth={liveWidth}
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
        </Svg>
      )}
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1 },
});
