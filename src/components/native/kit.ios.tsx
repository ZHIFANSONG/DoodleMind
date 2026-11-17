/**
 * 原生控件层（iOS 实现）。
 *
 * 这里**不是**用 RN 的 View 描一个看起来像系统的控件，而是把真正的 SwiftUI 视图
 * 塞进 RN 的视图树里：
 *
 *   RN View ──[Host]──▶ SwiftUI View
 *
 * `Host` 是 RN 到 SwiftUI 的桥；`matchContents` 让它把自己量出来的尺寸报回 Yoga，
 * 于是 SwiftUI 排好的控件在 RN 的流式布局里占一个正确的坑，而不是铺满整屏。
 *
 * 三个反复踩到的坑，写下来免得下次重踩：
 *
 * 1. `@expo/ui` 的每个 swift-ui 组件底层都是 `requireNativeView('ExpoUI', ...)`，
 *    **必须包在 `Host` 里**。挂在 Host 外面不会报错，只是什么都不显示。
 * 2. `Host` 不给 `matchContents` 时默认铺满父容器。放在一列 View 中间时，
 *    它会吃掉全部剩余高度，把下面的元素顶出屏幕。
 * 3. SwiftUI 的 `VStack` 默认按子视图的**理想宽度**排布，不是父容器宽度。
 *    要让卡片撑满，得在 `VStack` 上加 `frame({ maxWidth: 10000 })`
 *    （SwiftUI 里惯用的 `.infinity`，这里没有 Infinity 常量）。
 */

import {
  ContentUnavailableView,
  Gauge,
  Host,
  LabeledContent,
  Spacer,
  Text as NativeText,
  VStack,
  HStack,
} from '@expo/ui/swift-ui';
import {
  background,
  font,
  foregroundStyle,
  frame,
  gaugeStyle,
  padding,
  shapes,
  strokeBorder,
  opacity,
} from '@expo/ui/swift-ui/modifiers';

import { Palette, Radius } from '@/constants/design';
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
 * 宿主外壳。`matchContents` 是这里唯一重要的参数：
 * 它让 Host 只占 SwiftUI 内容那么大，从而能像普通 RN 元素一样参与流式布局。
 */
function Wrap({ children }: { children: React.ReactNode }) {
  return (
    <Host matchContents style={{ alignSelf: 'stretch' }}>
      {children}
    </Host>
  );
}

/** 卡片底：圆角 18 + 白底 + 1px 内描边。和 miniwatts 的 `Panel` 同一套参数。 */
const CARD = [
  padding({ all: 14 }),
  background(Palette.card, shapes.roundedRectangle({ cornerRadius: Radius.md, roundedCornerStyle: 'continuous' })),
  strokeBorder({
    content: Palette.rule,
    style: { lineWidth: 1 },
    shape: 'roundedRectangle',
    cornerRadius: Radius.md,
  }),
];

export function Panel({ title, trailing, children }: PanelProps) {
  return (
    <Wrap>
      <VStack alignment="leading" spacing={10} modifiers={[...CARD, frame({ maxWidth: 10000 })]}>
        {title || trailing ? (
          <HStack spacing={6} modifiers={[frame({ maxWidth: 10000 })]}>
            {title ? (
              <NativeText
                modifiers={[
                  font({ size: 10.5, weight: 'semibold', design: 'rounded' }),
                  foregroundStyle(Palette.inkFaint),
                ]}>
                {title.toUpperCase()}
              </NativeText>
            ) : null}
            <Spacer />
            {trailing ? (
              <NativeText
                modifiers={[font({ size: 10.5, design: 'monospaced' }), foregroundStyle(Palette.inkFaint)]}>
                {trailing}
              </NativeText>
            ) : null}
          </HStack>
        ) : null}
        {children}
      </VStack>
    </Wrap>
  );
}

export function Pill({ text, tint = Palette.inkFaint }: PillProps) {
  return (
    <Wrap>
      <NativeText
        modifiers={[
          font({ size: 11, weight: 'semibold' }),
          foregroundStyle(tint),
          padding({ horizontal: 9, vertical: 4 }),
          background(tint, shapes.capsule()),
          opacity(1),
        ]}>
        {text}
      </NativeText>
    </Wrap>
  );
}

export function Metric({ caption, value, unit, footnote, size = 24, tint = Palette.ink }: MetricProps) {
  const placeholder = value === '—';
  return (
    <Wrap>
      <VStack alignment="leading" spacing={3} modifiers={[frame({ maxWidth: 10000 })]}>
        <NativeText
          modifiers={[
            font({ size: 10.5, weight: 'semibold', design: 'rounded' }),
            foregroundStyle(Palette.inkFaint),
          ]}>
          {caption.toUpperCase()}
        </NativeText>
        <HStack spacing={3} alignment="firstTextBaseline">
          <NativeText
            modifiers={[
              font({ size: placeholder ? size * 0.7 : size, weight: 'bold', design: 'monospaced' }),
              foregroundStyle(placeholder ? Palette.inkFaint : tint),
            ]}>
            {value}
          </NativeText>
          {unit ? (
            <NativeText
              modifiers={[
                font({ size: size * 0.48, weight: 'medium', design: 'rounded' }),
                foregroundStyle(Palette.inkFaint),
              ]}>
              {unit}
            </NativeText>
          ) : null}
        </HStack>
        {footnote ? (
          <NativeText modifiers={[font({ size: 11, textStyle: 'caption2' }), foregroundStyle(Palette.inkFaint)]}>
            {footnote}
          </NativeText>
        ) : null}
      </VStack>
    </Wrap>
  );
}

/**
 * 线性表。
 *
 * 用系统的 `Gauge` + `linearCapacity` 样式，而不是自己画一条矩形：
 * 轨道、端点圆角、Dynamic Type、VoiceOver 的「进度」语义、以及 iOS 版本升级时
 * 外观跟着系统变——这些全部由 SwiftUI 提供。自己画的那条，等系统换样式那天就是孤岛。
 */
export function Bar({ label, detail, fraction, tint = Palette.ink }: GaugeProps) {
  const f = Math.max(0, Math.min(1, fraction));
  return (
    <Wrap>
      <VStack alignment="leading" spacing={5} modifiers={[frame({ maxWidth: 10000 })]}>
        {label || detail ? (
          <HStack spacing={6} alignment="firstTextBaseline" modifiers={[frame({ maxWidth: 10000 })]}>
            {label ? (
              <NativeText modifiers={[font({ size: 12.5 }), foregroundStyle(Palette.ink)]}>{label}</NativeText>
            ) : null}
            <Spacer />
            {detail ? (
              <NativeText
                modifiers={[font({ size: 13, weight: 'bold', design: 'monospaced' }), foregroundStyle(tint)]}>
                {detail}
              </NativeText>
            ) : null}
          </HStack>
        ) : null}
        <Gauge
          value={f}
          min={0}
          max={1}
          modifiers={[
            gaugeStyle('linearCapacity'),
            frame({ maxWidth: 10000 }),
            ...(tint === Palette.ink ? [] : []),
          ]}
        />
      </VStack>
    </Wrap>
  );
}

/** 环形表。`circularCapacity` 是系统的容量环，带中心读数。 */
export function Ring({ value, label, tint = Palette.ink }: RingProps) {
  const f = Math.max(0, Math.min(1, value));
  return (
    <Wrap>
      <Gauge
        value={f}
        min={0}
        max={1}
        modifiers={[gaugeStyle('circularCapacity'), ...(tint === Palette.ink ? [] : [])]}
        currentValueLabel={
          <NativeText modifiers={[font({ size: 15, weight: 'bold', design: 'monospaced' }), foregroundStyle(tint)]}>
            {label ?? `${Math.round(f * 100)}%`}
          </NativeText>
        }
      />
    </Wrap>
  );
}

/**
 * 规格表。用系统的 `LabeledContent`——左右两列，标签跟随 Dynamic Type，
 * VoiceOver 会读成「标签，值」而不是两个孤立的字符串。
 */
export function SpecPanel({ title, rows }: SpecPanelProps) {
  return (
    <Wrap>
      <VStack alignment="leading" spacing={3} modifiers={[...CARD, frame({ maxWidth: 10000 })]}>
        {title ? (
          <NativeText
            modifiers={[
              font({ size: 10.5, weight: 'semibold', design: 'rounded' }),
              foregroundStyle(Palette.inkFaint),
              padding({ bottom: 4 }),
            ]}>
            {title.toUpperCase()}
          </NativeText>
        ) : null}
        {rows.map((r) => (
          <LabeledContent key={r.label} label={r.label}>
            <NativeText
              modifiers={[font({ size: 11.5, design: 'monospaced' }), foregroundStyle(Palette.ink)]}>
              {r.value}
            </NativeText>
          </LabeledContent>
        ))}
      </VStack>
    </Wrap>
  );
}

/** 空态。iOS 17+ 的系统 `ContentUnavailableView`，和设置、App Store 里同一个控件。 */
export function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <Wrap>
      <ContentUnavailableView title={title} description={description} />
    </Wrap>
  );
}
