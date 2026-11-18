/**
 * 「它把哪些词搞混」的地图。
 *
 * 数据来自真实的误判：拿每类 24 张样本跑一遍，取 softmax 的平均，得到
 * `M[真值][它猜的]`；每类留下最容易混的 3 条边，用 Fruchterman-Reingold
 * 力导向布局把有边的节点拉到一起。所以**相邻 = 它真的会认错**，
 * 不是"权重方向相似"这种抽象说法。
 *
 * 走了两条弯路（都已写进 `tools/make_classmap.py`）：分类头权重 PCA 是一团均匀
 * 点云（解释方差 7%）；混淆画像的 MDS 保真度只有 6% —— 这套关系本来就是高维的，
 * 硬压成平面只会失真。力导向不声称保持全局距离，它只负责"连着的聚在一起"。
 *
 * 坐标离线算好打进 `assets/model/classmap.json`（30 KB），端上只画线画点。
 */
import { memo, useMemo } from 'react';
import Svg, { Circle, G, Line, Text as SvgText } from 'react-native-svg';
import { StyleSheet, View } from 'react-native';
import { Palette, Radius, Typography } from '@/constants/design';

type Point = { en: string; zh: string; emoji: string; x: number; y: number; mixed: string[] };
type Edge = { a: number; b: number; w: number };

const payload = require('../../assets/model/classmap.json') as { points: Point[]; edges: Edge[] };
const POINTS: Point[] = payload.points;
const EDGES: Edge[] = payload.edges ?? [];

/**
 * 地标词：148 个点如果全不标名字，读者没有参照物，图就只是一张蜘蛛网。
 * 挑 14 个最常见的涂鸦词当路牌——它们分布在图的各处，能把"这是哪儿"说清楚。
 * 自己不参与高亮逻辑，只在没有高亮时显示，免得和当前答案的名字打架。
 */
const LANDMARKS = [
  'sun',
  'moon',
  'cat',
  'fish',
  'house',
  'apple',
  'umbrella',
  'clock',
  'car',
  'bicycle',
  'bird',
  'tree',
  'mountain',
  'castle',
];

type Props = {
  /** 画布边长 */
  size: number;
  /** 要高亮的类别下标（一般是当前 top-3），空数组就是纯地图 */
  highlight?: number[];
  /** 显示多少个名字，太多会糊成一团 */
  labels?: number;
};

export const ClassMap = memo(function ClassMap({ size, highlight = [], labels = 0 }: Props) {
  const dots = useMemo(() => {
    const hi = new Map(highlight.map((idx, rank) => [idx, rank]));
    return POINTS.map((p, i) => ({
      ...p,
      i,
      rank: hi.get(i),
    }));
    // highlight 是数组，用 join 当依赖键（数组每次新建会让 memo 失效）
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [highlight.join(',')]);

  const hl = dots.filter((d) => d.rank !== undefined).sort((a, b) => (a.rank ?? 9) - (b.rank ?? 9));
  const labelSet = hl.slice(0, Math.max(labels, hl.length > 0 ? 3 : 0));
  // 有高亮时让位给答案，别再叠一圈路牌名字
  const landmarks = hl.length > 0 ? [] : dots.filter((d) => LANDMARKS.includes(d.en));

  return (
    <View style={[styles.wrap, { width: size, height: size }]}>
      <Svg width={size} height={size}>
        <G>
          {/* 先画边：两端都是"它会认错的一对"，越粗越容易混 */}
          {EDGES.map((e) => {
            const p = POINTS[e.a];
            const q = POINTS[e.b];
            if (!p || !q) return null;
            const touching = highlight.includes(e.a) || highlight.includes(e.b);
            return (
              <Line
                key={`e-${e.a}-${e.b}`}
                x1={p.x * size}
                y1={p.y * size}
                x2={q.x * size}
                y2={q.y * size}
                stroke={touching ? Palette.seal : Palette.inkFaint}
                strokeWidth={touching ? Math.max(0.8, e.w * 2.4) : Math.max(0.4, e.w * 1.2)}
                opacity={touching ? 0.5 : 0.16}
              />
            );
          })}
          {dots.map((d) => {
            const isHi = d.rank !== undefined;
            return (
              <Circle
                key={d.i}
                cx={d.x * size}
                cy={d.y * size}
                r={isHi ? Math.max(3.2, size * 0.011) : Math.max(1.2, size * 0.0034)}
                fill={isHi ? Palette.seal : Palette.inkSoft}
                opacity={isHi ? 1 : 0.5}
              />
            );
          })}
          {landmarks.map((d) => (
            <SvgText
              key={`lm-${d.i}`}
              x={d.x * size + 4}
              y={d.y * size - 4}
              fontSize={Math.max(7.5, size * 0.021)}
              fill={Palette.inkFaint}
              fontFamily={Typography.zh}>
              {d.zh}
            </SvgText>
          ))}
          {labelSet.map((d) => (
            <SvgText
              key={`l-${d.i}`}
              x={d.x * size + 6}
              y={d.y * size + 3}
              fontSize={Math.max(9, size * 0.028)}
              fill={Palette.seal}
              fontFamily={Typography.zh}>
              {d.zh}
            </SvgText>
          ))}
        </G>
      </Svg>
    </View>
  );
});

const styles = StyleSheet.create({
  wrap: {
    backgroundColor: Palette.card,
    borderRadius: Radius.md,
    borderWidth: 1,
    borderColor: Palette.rule,
    overflow: 'hidden',
  },
});
