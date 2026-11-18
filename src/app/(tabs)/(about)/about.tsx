/**
 * 关于：这一页不是「关于我们」，是**这个模型的体检报告**。
 *
 * 一个几百 KB 的神经网络装在一个没有网络的 App 里，玩家凭什么相信它没在唬人？
 * 答案只能是把它摊开：多少参数、多少类、一次推理多久、权重怎么存的、
 * 以及最关键的——**现场给它画一张，看它到底怎么想**。
 *
 * 最后那个可交互的实验台是这一页的全部理由。静态的规格表谁都能写，
 * 能当场验证的规格表才有说服力。
 */

import { useCallback, useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useScreenTop } from '@/hooks/use-screen-top';
import { Glyph } from '@/components/glyph';
import { Bar, SpecPanel } from '@/components/native/kit';
import { ClassMap } from '@/components/class-map';

import DrawingCanvas, { type Stroke } from '@/components/drawing-canvas';
import { BrushColors, Palette, Radius, Space, Typography } from '@/constants/design';
import { useDoodleInfer, useDoodleModel, useModelInfo } from '@/hooks/use-doodle-ai';

/**
 * 把 ops 列表压成一行人能读的结构描述。
 *
 * 相邻的 dw3x3 + pw1x1 合成一个 "dw+pw a→b"：它们在数学上就是一个可分离卷积块
 * （先逐通道卷积，再 1x1 混通道），分开写成两段反而让人以为网络更浅。
 * 池化点用 "↓" 标出来，因为读者最需要知道的是**特征图在哪一层开始变小**。
 */
function describeOps(ops: { type: string; cin: number; cout: number; pool: boolean }[]): string {
  if (ops.length === 0) return '—';
  const parts: string[] = [];
  for (let i = 0; i < ops.length; i++) {
    const op = ops[i];
    const next = ops[i + 1];
    if (op.type === 'dw3x3' && next && next.type === 'pw1x1') {
      parts.push(`dw+pw ${op.cin}→${next.cout}`);
      if (next.pool) parts.push('↓');
      i += 1;
      continue;
    }
    const name = op.type === 'dense3x3' ? 'conv3x3' : op.type === 'dw3x3' ? 'dw3x3' : 'pw1x1';
    parts.push(`${name} ${op.cin}→${op.cout}`);
    if (op.pool) parts.push('↓');
  }
  parts.push('→ GAP → FC');
  return parts.join(' ');
}

export default function AboutScreen() {
  const insets = useSafeAreaInsets();
  const padTop = useScreenTop();
  const { model, error, loadMs, classes } = useDoodleModel();
  const info = useModelInfo();
  const { run } = useDoodleInfer(model);

  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [probs, setProbs] = useState<Float32Array | null>(null);
  const [ms, setMs] = useState(0);
  const [board, setBoard] = useState(0);

  const infer = useCallback(
    (next: Stroke[]) => {
      setStrokes(next);
      const r = run(next, true);
      if (!r) return;
      setProbs(r.probs);
      setMs(r.inferMs);
    },
    [run],
  );

  const top = useMemo(() => {
    if (!probs) return [];
    return Array.from(probs, (p, index) => ({ index, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 8);
  }, [probs]);

  // 参数量与网络结构都从模型文件读（`assets/model/*.json` 的 `params` / `ops`）。
  // 早先在 TSX 里写死过 182,036 和 "conv 3x3 ×4"：换一次模型就得同步改文案，
  // 忘改就变成界面上印着一句假话——而这种假话没有任何报错，最省事的办法是别写死。
  const nParams = info.params;
  const archLabel = useMemo(() => describeOps(info.ops), [info.ops]);

  return (
    <ScrollView
      style={styles.root}
      contentContainerStyle={{
        paddingTop: padTop + Space.md,
        paddingBottom: insets.bottom + Space.xxl,
        paddingHorizontal: Space.lg,
      }}
      showsVerticalScrollIndicator={false}>
      {/* 标题「关于」由 UINavigationBar 的大标题画，这里不再写第二遍。 */}
      <Text style={styles.lede}>
        这是一个装在你手机里的小网络。它只有 {nParams.toLocaleString()} 个参数，
        {Math.round(info.bytes / 1024)} KB，不上网、不记录、不上传。它的全部世界观，来自 Google 公开的 5000 万张涂鸦。
      </Text>

      {/* 规格表。iOS 上是系统的 LabeledContent——Dynamic Type、VoiceOver
          「标签，值」的断句都交给 UIKit，不是我们拿两列 Text 硬凑。 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>规格</Text>
        <SpecPanel title="MODEL" rows={[
          { label: '类别数', value: `${info.nClasses} 类` },
          { label: '参数量', value: nParams.toLocaleString() },
          { label: '权重体积', value: `${Math.round(info.bytes / 1024)} KB` },
          { label: '网络结构', value: archLabel },
          { label: '输入', value: '32×32 灰度，笔画半径 1px' },
          { label: '量化', value: '权重 per-channel int8，激活 float32' },
          { label: '推理', value: `${ms > 0 ? ms : '—'} ms（本机实测）` },
          { label: '唤醒', value: `${loadMs} ms` },
          { label: '网络权限', value: '无。App 里没有一行网络代码' },
        ]} />
      </View>

      {/* 现场试 */}
      <Section title="现场试">
        <Text style={styles.sectionNote}>
          随便画点什么。下面就是它此刻的完整想法——不是置信度最高的那一个，是全部 {info.nClasses} 类的排名。
        </Text>
        <View
          style={styles.board}
          onLayout={(e) => setBoard(Math.floor(e.nativeEvent.layout.width))}>
          {board > 0 && (
            <DrawingCanvas
              strokes={strokes}
              color={BrushColors[0]}
              enabled={!!model}
              onStrokeEnd={infer}
              onPoint={(s) => {
                if (s.length % 2 === 0) infer(s);
              }}
            />
          )}
          {!model && !error && <Text style={styles.boardHint}>正在唤醒模型…</Text>}
        </View>

        <View style={styles.bars}>
          {top.length === 0 && <Text style={styles.boardHint}>画两笔，这里就活了</Text>}
          {top.length > 0 && (
            // 榜首用 SwiftUI 的真 Gauge（linearCapacity）：轨道、动画、Dynamic Type、
            // VoiceOver 的「进度」语义都交给系统。其余名次用 RN 细条——
            // 每多一个 Gauge 就多一次跨 RN→SwiftUI 边界，9 条不值得。
            <Bar
              label={`它最确信：${classes[top[0].index]?.zh ?? ''}`}
              detail={`${Math.round(top[0].p * 100)}%`}
              fraction={top[0].p}
            />
          )}
          {top.slice(1).map((t) => {
            const c = classes[t.index];
            return (
              <View key={t.index} style={styles.barRow}>
                <Text style={styles.barWord} numberOfLines={1}>
                  {c?.emoji} {c?.zh}
                </Text>
                <View style={styles.barTrack}>
                  <View style={[styles.barFill, { width: `${Math.max(1, t.p * 100)}%` }]} />
                </View>
                <Text style={styles.barPct}>{(t.p * 100).toFixed(1)}%</Text>
              </View>
            );
          })}
        </View>
        {strokes.length > 0 && (
          <Pressable
            onPress={() => {
              setStrokes([]);
              setProbs(null);
            }}
            style={styles.resetBtn}>
            <Text style={styles.resetBtnText}>擦掉重画</Text>
          </Pressable>
        )}
      </Section>

      {/* 它眼里的世界 */}
      <Section title="它把哪些词搞混">
        <Text style={styles.sectionNote}>
          每类留 24 张样本跑一遍，看它平均会猜成什么；最容易混的三条连成线，再用力导向布局把有瓜葛的词拉到一起。
          所以挨着就代表它真的认错过。上面画的时候，它的前几名会在这里亮起来。
        </Text>
        {board > 0 && (
          <ClassMap size={board} highlight={top.slice(0, 3).map((t) => t.index)} />
        )}
        <Text style={styles.sectionNote}>
          这张图解释了两件事：为什么画猫会被认成老虎（同一簇），以及它其实没有「类别相似」的概念——
          只有「这团像素长得像哪一类的训练样本」。坐标由
          <Text style={styles.code}> tools/make_classmap.py </Text>
          离线算好，端上只画线画点。
        </Text>
      </Section>

      {/* 设计取舍 */}
      <Section title="为什么这么做">
        <Note
          head="为什么不用 TensorFlow.js"
          body={
            'tfjs 的 wasm 后端在 Expo Go 里跑不起来（需要 WASM 支持），纯 JS 后端比这份手写代码慢 3~5 倍，' +
            '还得多两兆依赖。这个网络的形状是固定的（4 段 conv，全是 3x3），手写 forward 只有 250 行，' +
            '能复用两块 Float32Array 缓冲做到零分配——这是 tfjs 的通用算子调度做不到的。'
          }
        />
        <Note
          head="为什么权重存 int8，但激活不量化"
          body={
            '权重 int8 把 fp32 的几 MB 压缩到这个体积，一次解量化后常驻内存，推理时不重复付代价。' +
            '激活不量化，是因为 JS 里没有 SIMD，整数乘加的吞吐并不比浮点高，' +
            '而 4 段 conv 的激活量化误差会一层层累积，实测掉好几个点准确率。' +
            '换句话说：量化只用在真的省资源的地方。'
          }
        />
        <Note
          head="为什么判定要「概率 + 排名 + 保持 380ms」三条同时成立"
          body={
            '单帧的 top-1 会剧烈抖动——同一张图多画一笔，排名就可能翻转。' +
            '如果只看单帧命中，玩家体验会在「明明画对了却不给过」和「随便两笔就过了」之间横跳。' +
            '加一层 EMA 平滑，再要求连续保持 380 毫秒，手感才稳。' +
            '这 380ms 也是刻意的：它能被感知（有等待感），但不长到令人烦躁。'
          }
        />
        <Note
          head="为什么画布必须正方形"
          body={
            '模型输入 32×32，训练数据也全是正方形构图。画布做成非正方形，栅格化时按包围盒等比缩放' +
            '会把内容压扁，模型判断会悄悄变差——这种 bug 不报错，只会让人觉得「这 AI 有点笨」。'
          }
        />
      </Section>

      {/* 模型看不到什么 */}
      <Section title="它看不见什么">
        <Note
          head="颜色"
          body="输入是单通道灰度。你用红色画还是黑色画，对它来说是同一张图。"
        />
        <Note
          head="你的手速"
          body="笔迹的粗细变化只存在于显示层，栅格化时被抹平——训练数据里也没有压感信息。"
        />
        <Note
          head="画的顺序之外的意图"
          body="它只看最终那一张 32×32。你画一只猫想表达「可爱」，它只知道像素。"
        />
      </Section>

      <Section title="数据来源">
        <Note
          head="Google QuickDraw"
          body={
            '345 个类别、5000 万张手绘涂鸦，CC BY 4.0。本 App 取了其中 148 类，每类 4500 张，' +
            '共 66.6 万张。数据全部在本机用 Python 栅格化后训练，训练脚本在仓库里。' +
            'App 本身不含任何数据集，只带一份权重（体积见上方规格表）。'
          }
        />
      </Section>

      <View style={styles.footer}>
        <View style={styles.footerDot} />
        <Text style={styles.footerText}>
          DoodleMind · 手写 TS 推理 · 无网络代码 · MIT License
        </Text>
      </View>
    </ScrollView>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      <View style={styles.sectionBody}>{children}</View>
    </View>
  );
}

function Note({ head, body }: { head: string; body: string }) {
  return (
    <View style={styles.note}>
      <View style={styles.noteHead}>
        <Glyph name="dot" color={Palette.seal} size={5} />
        <Text style={styles.noteHeadText}>{head}</Text>
      </View>
      <Text style={styles.noteBody}>{body}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Palette.paper },
  title: { color: Palette.ink, fontSize: 24, fontWeight: '700', fontFamily: Typography.title },
  lede: {
    color: Palette.inkSoft,
    fontSize: 13,
    lineHeight: 21,
    fontFamily: Typography.zh,
    marginTop: Space.sm,
  },

  section: { marginTop: Space.xl, gap: Space.sm },
  sectionTitle: {
    color: Palette.ink,
    fontSize: 13,
    fontWeight: '700',
    fontFamily: Typography.zh,
    letterSpacing: 0.4,
  },
  sectionBody: {
    backgroundColor: Palette.card,
    borderRadius: Radius.md,
    borderWidth: 1,
    borderColor: Palette.rule,
    padding: Space.md,
    gap: Space.xs,
  },
  sectionNote: { color: Palette.inkFaint, fontSize: 11.5, lineHeight: 17, fontFamily: Typography.zh },
  code: { fontFamily: Typography.mono, fontSize: 11 },

  board: {
    aspectRatio: 1,
    width: '100%',
    borderRadius: Radius.sm,
    backgroundColor: Palette.paper,
    borderWidth: 1,
    borderColor: Palette.rule,
    overflow: 'hidden',
    marginTop: Space.xs,
  },
  boardHint: { color: Palette.inkFaint, fontSize: 12, fontFamily: Typography.zh, textAlign: 'center', paddingVertical: 8 },

  bars: { gap: 4, marginTop: Space.sm },
  barRow: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  barWord: { width: 84, color: Palette.ink, fontSize: 11.5, fontFamily: Typography.zh },
  barTrack: {
    flex: 1,
    height: 7,
    borderRadius: 4,
    backgroundColor: Palette.paperSunken,
    overflow: 'hidden',
  },
  barFill: { height: 7, backgroundColor: Palette.ink, borderRadius: 4 },
  barPct: { width: 40, textAlign: 'right', color: Palette.inkFaint, fontSize: 10, fontFamily: Typography.mono },

  resetBtn: { alignSelf: 'flex-start', marginTop: Space.xs },
  resetBtnText: { color: Palette.seal, fontSize: 11.5, fontFamily: Typography.zh, fontWeight: '600' },

  note: { gap: 3, paddingVertical: 5 },
  noteHead: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  noteHeadText: { color: Palette.ink, fontSize: 12, fontWeight: '600', fontFamily: Typography.zh },
  noteBody: { color: Palette.inkSoft, fontSize: 11.5, lineHeight: 18, fontFamily: Typography.zh },

  footer: { flexDirection: 'row', alignItems: 'center', gap: 7, marginTop: Space.xxl },
  footerDot: { width: 9, height: 9, borderRadius: 5, backgroundColor: Palette.ink },
  footerText: { color: Palette.inkFaint, fontSize: 10.5, fontFamily: Typography.zh },
});
