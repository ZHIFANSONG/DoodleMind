/**
 * 关于 / 技术说明。
 *
 * 这一页不是为了自夸，是为了让来看源码的人（和三个月后的我）
 * 一眼看懂三件事：模型怎么来的、为什么不用 TF.js、数据有没有出过手机。
 * 顺带做一个可交互的调试区：把任意涂鸦实时喂给模型，直接看 148 类的完整分布。
 */

import { useCallback, useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { SymbolView } from 'expo-symbols';

import DrawingCanvas, { type Stroke } from '@/components/drawing-canvas';
import { Palette, Radius, Space, Typography } from '@/constants/design';
import { useDoodleInfer, useDoodleModel, useModelInfo } from '@/hooks/use-doodle-ai';

const FACTS = [
  ['类别', '148 个你画我猜词，中英双语 + emoji'],
  ['输入', '32×32 灰度，笔画按圆心距离抗锯齿栅格化'],
  ['网络', '4 段卷积（24/48/96/128）→ 全局平均池化 → 全连接'],
  ['权重量化', '每输出通道对称 int8，激活保持 fp32'],
  ['推理', '纯 TypeScript 手写，无 TF.js / 无 WebGL / 无 Wasm'],
  ['数据', 'Google QuickDraw simplified，端上只跑推理'],
  ['网络权限', '一个都不用。App 全程离线'],
];

const WHY = [
  {
    title: '为什么不用 TensorFlow.js',
    body:
      'tfjs 的 React Native 后端要么走 WebView（多一层桥，冷启动 300ms+），要么依赖尚未稳定的原生绑定。' +
      '我们这张网只有 4 层卷积、16 万参数，手写 300 行前向传播跑一次 3–8ms，比装一整个框架快得多。' +
      '代价是要自己写 im2col 和 BN 折叠——好处是全部可控、可调试、包体只多 160KB。',
  },
  {
    title: '为什么权重量化到 int8，但不量化激活',
    body:
      'JS 没有 SIMD，整数运算并不比浮点快，所以量化省的不是算力而是**包体和内存**：' +
      '650KB fp32 → 163KB int8（base64 后仍在 220KB 以内）。' +
      '而激活量化会在 4 段卷积上累积量化误差，实测掉 3–5 个百分点准确率，不划算。' +
      '于是折中：权重存 int8，加载时解量化成 Float32Array 常驻，推理全程 fp32。',
  },
  {
    title: '为什么判定要「双条件 + 保持时长」',
    body:
      '单帧 top-1 极不稳定：同一张图多画一笔，第一名就可能换人。' +
      '如果只看单帧，玩家会觉得「我明明画对了它却不给过」。' +
      '所以要求目标类概率经 EMA 平滑后既过阈值、又排第一，还得连续保持 380ms 才结算。' +
      '这三条同时满足时，玩家看到的进度条会先涨满——过与不过都变得可预期。',
  },
  {
    title: '为什么画布必须是正方形',
    body:
      '模型输入是正方形，QuickDraw 原始数据也是正方形构图。' +
      '画布一旦不是 1:1，栅格化时按包围盒等比缩放就会把内容压扁，' +
      '而模型对这种形变非常敏感——准确率会悄悄掉，且极难察觉。锁定 1:1 省掉一整类 bug。',
  },
];

export default function AboutScreen() {
  const insets = useSafeAreaInsets();
  const { model, error, classes } = useDoodleModel();
  const info = useModelInfo();
  const { run } = useDoodleInfer(model);

  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [probs, setProbs] = useState<Float32Array | null>(null);
  const [ms, setMs] = useState(0);

  const onStrokeEnd = useCallback(
    (next: Stroke[]) => {
      setStrokes(next);
      const res = run(next, true);
      if (!res) return;
      setProbs(res.probs);
      setMs(res.inferMs);
    },
    [run],
  );

  // 展示完整 148 类分布里最高的 10 个
  const top = useMemo(() => {
    if (!probs) return [];
    return Array.from(probs, (p, index) => ({ index, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 10);
  }, [probs]);

  const weightKb = Math.round(info.bytes / 1024);

  return (
    <ScrollView
      style={styles.root}
      contentContainerStyle={{ paddingTop: insets.top + Space.sm, paddingBottom: insets.bottom + 90 }}>
      <Text style={styles.h1}>DoodleMind</Text>
      <Text style={styles.lede}>
        一个完全离线的手绘识别模型。你画，它猜，全程不出手机。
      </Text>

      <Section title="模型规格">
        <View style={styles.factTable}>
          {FACTS.map(([k, v]) => (
            <View key={k} style={styles.factRow}>
              <Text style={styles.factKey}>{k}</Text>
              <Text style={styles.factVal}>{v}</Text>
            </View>
          ))}
          <View style={styles.factRow}>
            <Text style={styles.factKey}>实测体积</Text>
            <Text style={styles.factVal}>
              {weightKb} KB 权重 · {info.nClasses} 类 · 通道 {info.width.join('/')}
            </Text>
          </View>
        </View>
      </Section>

      <Section title="试一下" subtitle="在下面随便画，直接看它在你手上的真实延迟">
        <View style={styles.debugCanvas}>
          <DrawingCanvas strokes={strokes} color={Palette.ink} enabled={!!model} onStrokeEnd={onStrokeEnd}>
            {!model && !error && <Text style={styles.debugHint}>模型加载中…</Text>}
            {model && !probs && <Text style={styles.debugHint}>画点什么</Text>}
          </DrawingCanvas>
        </View>
        <View style={styles.debugBar}>
          <Text style={styles.debugMs}>{ms > 0 ? `${ms} ms / 次` : '待测'}</Text>
          <Pressable
            onPress={() => {
              setStrokes([]);
              setProbs(null);
              setMs(0);
            }}
            hitSlop={8}>
            <Text style={styles.debugClear}>清空</Text>
          </Pressable>
        </View>

        {top.map((t, i) => (
          <View key={t.index} style={styles.barRow}>
            <Text style={styles.barRank}>{i + 1}</Text>
            <Text style={styles.barWord} numberOfLines={1}>
              {classes[t.index]?.emoji} {classes[t.index]?.zh}
            </Text>
            <View style={styles.barTrack}>
              <View style={[styles.barFill, { width: `${Math.round(t.p * 100)}%` }]} />
            </View>
            <Text style={styles.barPct}>{(t.p * 100).toFixed(1)}%</Text>
          </View>
        ))}
        {error && <Text style={styles.err}>{error}</Text>}
      </Section>

      <Section title="设计取舍">
        {WHY.map((w) => (
          <View key={w.title} style={styles.whyCard}>
            <View style={styles.whyHead}>
              <SymbolView name="chevron.right" tintColor={Palette.seal} size={12} resizeMode="scaleAspectFit" />
              <Text style={styles.whyTitle}>{w.title}</Text>
            </View>
            <Text style={styles.whyBody}>{w.body}</Text>
          </View>
        ))}
      </Section>

      <Section title="致谢">
        <Text style={styles.thanks}>
          QuickDraw 数据集来自 Google Creative Lab（CC BY 4.0）。{'\n'}
          笔画平滑用的是 Catmull-Rom 样条，栅格化用的是圆头笔距离场。{'\n'}
          整个 App 不含任何网络代码。
        </Text>
      </Section>
    </ScrollView>
  );
}

function Section({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {subtitle && <Text style={styles.sectionSub}>{subtitle}</Text>}
      <View style={{ marginTop: Space.sm }}>{children}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Palette.paper, paddingHorizontal: Space.lg },
  h1: { color: Palette.ink, fontSize: 32, fontWeight: '700', fontFamily: Typography.zh },
  lede: {
    color: Palette.inkSoft,
    fontSize: 13,
    fontFamily: Typography.zh,
    lineHeight: 21,
    marginTop: 4,
  },

  section: { marginTop: Space.xl },
  sectionTitle: { color: Palette.ink, fontSize: 17, fontWeight: '700', fontFamily: Typography.zh },
  sectionSub: { color: Palette.inkFaint, fontSize: 12, fontFamily: Typography.zh, marginTop: 2 },

  factTable: {
    backgroundColor: Palette.card,
    borderRadius: Radius.md,
    borderWidth: 1,
    borderColor: Palette.rule,
    paddingHorizontal: Space.md,
  },
  factRow: {
    flexDirection: 'row',
    paddingVertical: 9,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: Palette.rule,
    gap: Space.sm,
  },
  factKey: { width: 68, color: Palette.inkFaint, fontSize: 12, fontFamily: Typography.zh },
  factVal: { flex: 1, color: Palette.ink, fontSize: 12, fontFamily: Typography.zh, lineHeight: 18 },

  debugCanvas: {
    aspectRatio: 1,
    borderRadius: Radius.md,
    backgroundColor: Palette.card,
    borderWidth: 1,
    borderColor: Palette.rule,
    overflow: 'hidden',
  },
  debugHint: {
    position: 'absolute',
    alignSelf: 'center',
    top: '46%',
    color: Palette.inkFaint,
    fontSize: 12,
    fontFamily: Typography.zh,
  },
  debugBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 7,
  },
  debugMs: { color: Palette.inkSoft, fontSize: 12, fontFamily: Typography.mono },
  debugClear: { color: Palette.seal, fontSize: 12, fontFamily: Typography.zh, fontWeight: '600' },

  barRow: { flexDirection: 'row', alignItems: 'center', gap: 7, paddingVertical: 3 },
  barRank: { width: 16, color: Palette.inkFaint, fontSize: 10, fontFamily: Typography.mono },
  barWord: { width: 84, color: Palette.ink, fontSize: 12, fontFamily: Typography.zh },
  barTrack: {
    flex: 1,
    height: 6,
    borderRadius: 3,
    backgroundColor: Palette.paperSunken,
    overflow: 'hidden',
  },
  barFill: { height: 6, borderRadius: 3, backgroundColor: Palette.indigo },
  barPct: { width: 44, textAlign: 'right', color: Palette.inkSoft, fontSize: 10, fontFamily: Typography.mono },

  whyCard: {
    backgroundColor: Palette.card,
    borderRadius: Radius.md,
    borderWidth: 1,
    borderColor: Palette.rule,
    padding: Space.md,
    marginBottom: Space.sm,
  },
  whyHead: { flexDirection: 'row', alignItems: 'center', gap: 5, marginBottom: 5 },
  whyTitle: { color: Palette.ink, fontSize: 13, fontWeight: '700', fontFamily: Typography.zh },
  whyBody: { color: Palette.inkSoft, fontSize: 12, fontFamily: Typography.zh, lineHeight: 20 },

  thanks: { color: Palette.inkFaint, fontSize: 12, fontFamily: Typography.zh, lineHeight: 21 },
  err: { color: Palette.seal, fontSize: 11, fontFamily: Typography.mono, marginTop: Space.sm },
});
