/**
 * 判读区：这一局最该看的读数。
 *
 * 为什么单独抽成一个 `memo` 组件，而不是留在 `play.tsx` 里直接写：
 * 游戏页的计时器每 33ms 触发一次整页重渲染。判读区里的置信度条在 iOS 上是
 * **真的 SwiftUI `Gauge`**（跨 RN → SwiftUI 边界），每一次重渲染都要重新
 * 序列化一遍 props 交给原生侧。如果不做记忆化，一秒就有 30 次无意义的
 * 跨边界调用——而实际上百分比一秒才变一两次。
 *
 * 所以 props 全部压成**原始值**（数字/字符串），让 `React.memo` 的浅比较
 * 真的能挡住绝大多数帧：
 * - `pct` 传整数百分比，不是 0..1 的浮点（浮点几乎每帧都在动）
 * - `secs` 传整秒
 * - `ink` 传取整后的墨量
 *
 * 顺带一个设计上的理由：移动端和 web 用的是**同一个组件**，只是 `kit.tsx`
 * 与 `kit.ios.tsx` 两份实现。判读区的语义（哪句话、什么颜色、什么时候变红）
 * 只有一份，不会出现「web 上看着对、真机上文案不一样」。
 */

import { memo } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { Bar } from '@/components/native/kit';
import { Palette, Space, Typography } from '@/constants/design';
import type { Mode } from '@/lib/modes';

export type PlayMeterProps = {
  mode: Mode;
  /** 目标类的置信度，取整后的百分数（0..100）。 */
  pct: number;
  /** 目标类当前的排名，0 = 第一。`< 0` 表示没有目标。 */
  rank: number;
  strokeCount: number;
  /** 已用整秒。 */
  secs: number;
  /** 取整后的墨量（等效满墨像素数）。 */
  ink: number;
  /** 盲区模式的耍赖线。 */
  minInk: number;
  hint: string | null;
};

function PlayMeterImpl({
  mode,
  pct,
  rank,
  strokeCount,
  secs,
  ink,
  minInk,
  hint,
}: PlayMeterProps) {
  const blind = mode === 'blind';
  const free = mode === 'free';
  const hasTarget = !free;
  const inkEnough = ink >= minInk;

  return (
    <View style={styles.block}>
      {hasTarget ? (
        <>
          <Bar
            label={blind ? '它有多确定你画的这个词' : '它有多确定这是'}
            detail={`${pct}%`}
            fraction={pct / 100}
            tint={blind ? Palette.seal : Palette.ink}
          />
          <Text style={styles.note}>{noteFor(strokeCount, rank)}</Text>
        </>
      ) : (
        <Text style={styles.label}>没有目标——它看到什么就说什么</Text>
      )}

      <View style={styles.readout}>
        <Text style={styles.item}>{strokeCount} 笔</Text>
        <Text style={styles.dot}>·</Text>
        <Text style={styles.item}>{secs}s</Text>
        <Text style={styles.dot}>·</Text>
        <Text style={styles.item}>墨 {ink}</Text>
        {blind ? (
          <Text style={[styles.item, inkEnough && styles.ok]}>
            {inkEnough ? '· 够数了' : `· 还差 ${Math.max(0, minInk - ink)} 才算数`}
          </Text>
        ) : null}
      </View>

      {hint ? <Text style={styles.hint}>{hint}</Text> : null}
    </View>
  );
}

/**
 * 三种「排名」的说法必须分开，这是我在真机上撞出来的：
 * 追踪器还没喂过任何一帧时，148 个概率全是 0，排序后目标恰好落在第 42 位。
 * 直接把 rank+1 印出来，空画布上会显示「现在它排第 42 位」——纯属胡说。
 */
function noteFor(strokeCount: number, rank: number): string {
  if (strokeCount === 0) return '还没落笔，它只能瞎猜';
  if (rank <= 0) return '它已经把你画的当成它了';
  return `现在它排第 ${rank + 1} 位`;
}

export const PlayMeter = memo(PlayMeterImpl);

const styles = StyleSheet.create({
  block: { paddingHorizontal: 2, marginTop: Space.md, gap: 6 },
  label: { color: Palette.inkFaint, fontSize: 11, fontFamily: Typography.zh },
  note: { color: Palette.inkFaint, fontSize: 10.5, fontFamily: Typography.zh },

  readout: { flexDirection: 'row', gap: 5, flexWrap: 'wrap' },
  item: { color: Palette.inkFaint, fontSize: 10.5, fontFamily: Typography.mono },
  dot: { color: Palette.rule, fontSize: 10.5 },
  ok: { color: Palette.jade, fontWeight: '700' },

  hint: {
    color: Palette.inkSoft,
    fontSize: 11.5,
    lineHeight: 16,
    fontFamily: Typography.zh,
    paddingHorizontal: 2,
  },
});
