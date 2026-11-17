/**
 * 原生控件层（web / Android 回退实现）。
 *
 * iOS 上同名的 `kit.ios.tsx` 会用 `@expo/ui/swift-ui` 直接创建真正的 SwiftUI 视图
 * （`Host` / `LabeledContent` / `Gauge` / `ContentUnavailableView` /
 * `glassEffect`），而不是用 RN 的 View 描一个长得像的。
 *
 * 这个文件存在的唯一理由是：同一个页面在 web 上也要能看、在 Android 上不能崩。
 * 所以对外 API 必须与 `kit.ios.tsx` 严格一致——两边任何一边多一个 prop，
 * TypeScript 只有在 import 到边界的那个文件时才看得见，很容易漏。
 * 所以 props 类型统一放在 `kit-types.ts`，两边都 import 它。
 */

import { StyleSheet, Text, View } from 'react-native';

import { Palette, Radius, Space, Typography } from '@/constants/design';
import type {
  EmptyStateProps,
  GaugeProps,
  MetricProps,
  PanelProps,
  PillProps,
  RingProps,
  SpecPanelProps,
} from '@/components/native/kit-types';

/**
 * 带标题的卡片。对齐 miniwatts 的 `Panel`：标题是**全大写 + 字距撑开的小字**，
 * 右上角挂一个等宽的补充读数（`trailing`）。这套排版让一屏里摞三张卡也不乱。
 */
export function Panel({ title, trailing, children }: PanelProps) {
  return (
    <View style={s.panel}>
      {(title || trailing) && (
        <View style={s.panelHead}>
          {title ? <Text style={s.panelTitle}>{title.toUpperCase()}</Text> : null}
          <View style={{ flex: 1 }} />
          {trailing ? <Text style={s.panelTrailing}>{trailing}</Text> : null}
        </View>
      )}
      {children}
    </View>
  );
}

/** 状态药丸。`tint` 决定文字与底色，底色是 tint 的 14% 透明度。 */
export function Pill({ text, tint = Palette.inkFaint }: PillProps) {
  return (
    <View style={[s.pill, { backgroundColor: hexA(tint, 0.14), borderColor: hexA(tint, 0.3) }]}>
      <Text style={[s.pillText, { color: tint }]}>{text}</Text>
    </View>
  );
}

/**
 * 读数。caption / 大数字 + 单位 / 脚注。数字用等宽数字（`fontVariant`），
 * 否则每次刷新宽度都会变，整行会左右抖。
 */
export function Metric({ caption, value, unit, footnote, size = 24, tint = Palette.ink }: MetricProps) {
  const placeholder = value === '—';
  return (
    <View style={s.metric}>
      <Text style={s.metricCaption}>{caption.toUpperCase()}</Text>
      <View style={s.metricRow}>
        <Text
          style={[
            s.metricValue,
            { fontSize: placeholder ? size * 0.7 : size, color: placeholder ? Palette.inkFaint : tint },
          ]}>
          {value}
        </Text>
        {unit ? <Text style={[s.metricUnit, { fontSize: size * 0.48 }]}>{unit}</Text> : null}
      </View>
      {footnote ? <Text style={s.metricFoot}>{footnote}</Text> : null}
    </View>
  );
}

/**
 * 线性表。`label` 左、读数右、下面一条 5px 的胶囊轨。
 * 用于「置信度」「墨量够不够」这类 0..1 的量。
 */
export function Bar({ label, detail, fraction, tint = Palette.ink }: GaugeProps) {
  const f = Math.max(0, Math.min(1, fraction));
  return (
    <View style={s.bar}>
      <View style={s.barHead}>
        <Text style={s.barLabel} numberOfLines={1}>
          {label}
        </Text>
        <View style={{ flex: 1 }} />
        <Text style={[s.barDetail, { color: tint }]}>{detail}</Text>
      </View>
      <View style={s.barTrack}>
        <View style={[s.barFill, { width: `${f * 100}%`, backgroundColor: tint }]} />
      </View>
    </View>
  );
}

/** 大号环形表。miniwatts 的 `PowerRing` 是靠 Canvas 画刻度弧；这里退回线性表。 */
export function Ring({ value, label, tint = Palette.ink }: RingProps) {
  return <Bar label={label ?? ''} detail={`${Math.round(value * 100)}%`} fraction={value} tint={tint} />;
}

export function SpecPanel({ title, rows }: SpecPanelProps) {
  return (
    <View style={s.specPanel}>
      {title ? <Text style={s.panelTitle}>{title.toUpperCase()}</Text> : null}
      {rows.map((r) => (
        <View key={r.label} style={s.specRow}>
          <Text style={s.specK}>{r.label}</Text>
          <Text style={s.specV}>{r.value}</Text>
        </View>
      ))}
    </View>
  );
}

export function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <View style={s.empty}>
      <Text style={s.emptyTitle}>{title}</Text>
      {description ? <Text style={s.emptyBody}>{description}</Text> : null}
    </View>
  );
}

function hexA(hex: string, alpha: number): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

const s = StyleSheet.create({
  panel: {
    backgroundColor: Palette.card,
    borderRadius: Radius.md,
    borderWidth: 1,
    borderColor: Palette.rule,
    padding: Space.md,
    gap: 10,
  },
  panelHead: { flexDirection: 'row', alignItems: 'center' },
  panelTitle: {
    color: Palette.inkFaint,
    fontSize: 10,
    letterSpacing: 1.1,
    fontWeight: '700',
    fontFamily: Typography.zh,
  },
  panelTrailing: { color: Palette.inkFaint, fontSize: 10.5, fontFamily: Typography.mono },

  pill: {
    alignSelf: 'flex-start',
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 9,
    paddingVertical: 4,
    borderRadius: Radius.pill,
    borderWidth: 0.5,
  },
  pillText: { fontSize: 11, fontWeight: '600', fontFamily: Typography.zh },

  metric: { gap: 2 },
  metricCaption: {
    color: Palette.inkFaint,
    fontSize: 10,
    letterSpacing: 1,
    fontWeight: '700',
    fontFamily: Typography.zh,
  },
  metricRow: { flexDirection: 'row', alignItems: 'baseline', gap: 3 },
  metricValue: { fontWeight: '700', fontFamily: Typography.mono, fontVariant: ['tabular-nums'] },
  metricUnit: { color: Palette.inkFaint, fontWeight: '600', fontFamily: Typography.mono },
  metricFoot: { color: Palette.inkFaint, fontSize: 10.5, fontFamily: Typography.zh, lineHeight: 15 },

  bar: { gap: 5 },
  barHead: { flexDirection: 'row', alignItems: 'baseline' },
  barLabel: { color: Palette.ink, fontSize: 12.5, fontFamily: Typography.zh },
  barDetail: { fontSize: 13, fontWeight: '700', fontFamily: Typography.mono, fontVariant: ['tabular-nums'] },
  barTrack: { height: 5, borderRadius: 3, backgroundColor: Palette.paperSunken, overflow: 'hidden' },
  barFill: { height: 5, borderRadius: 3 },

  specPanel: {
    backgroundColor: Palette.card,
    borderRadius: Radius.md,
    borderWidth: 1,
    borderColor: Palette.rule,
    padding: Space.md,
    gap: 2,
  },
  specRow: { flexDirection: 'row', alignItems: 'baseline', gap: Space.sm, paddingVertical: 3 },
  specK: { color: Palette.inkFaint, fontSize: 11.5, fontFamily: Typography.zh, width: 76 },
  specV: { flex: 1, color: Palette.ink, fontSize: 11.5, fontFamily: Typography.mono },

  empty: { alignItems: 'center', gap: Space.xs, paddingVertical: Space.xl },
  emptyTitle: { color: Palette.ink, fontSize: 15, fontWeight: '600', fontFamily: Typography.zh },
  emptyBody: { color: Palette.inkFaint, fontSize: 12, fontFamily: Typography.zh },
});
