/**
 * 原生控件层的共享契约。
 *
 * 放在独立文件里，是因为 `kit.tsx`（web/Android）与 `kit.ios.tsx`（SwiftUI）
 * 是两份实现，Metro 按平台只加载其中一份——**TypeScript 也只看得到一份**。
 * 如果 props 类型写在某一侧，另一侧写错了 tsc 永远发现不了，
 * 只会在真机上少一个 prop、少一个回调，而且不报错。
 *
 * 所以类型住在这里，两侧都 import 它。
 */

import type { ReactNode } from 'react';

import type { Palette } from '@/constants/design';

/** 卡片里的一行规格。`value` 是可以带单位/百分比的成品字符串。 */
export type SpecRow = { label: string; value: string };

/** 底部导航的语义色。只允许调色板里的名字，避免两侧各写一个十六进制。 */
export type Tint = (typeof Palette)[keyof typeof Palette];

export type PanelProps = {
  /** 卡片的眉题。iOS 上由 `mwCaption` 风格绘制（全大写 + 字距）。 */
  title?: string;
  /** 右上角的补充读数，等宽小字。 */
  trailing?: string;
  children: ReactNode;
};

export type PillProps = {
  text: string;
  tint?: string;
};

export type MetricProps = {
  caption: string;
  /** 已成型的读数字符串；`'—'` 表示暂无数据（会缩小并淡化）。 */
  value: string;
  unit?: string;
  footnote?: string;
  /** 主数字字号，默认 24。 */
  size?: number;
  tint?: string;
};

/**
 * 0..1 的线性表。
 * `label` 是左边的说明，`detail` 是右边那个已经格式化好的读数——
 * 之所以让调用方传字符串而不是自己算百分比，是因为不同场景要的写法不一样
 * （「63%」「第 1 位」「还差 12」），在这里统一只会逼出一堆开关。
 */
export type GaugeProps = {
  label?: string;
  detail?: string;
  fraction: number;
  tint?: string;
};

/** 环形表：`value` 是 0..1 的比值本身（不是已经乘过 100 的），由控件自己排版。 */
export type RingProps = { value: number; label?: string; tint?: string };

export type SpecPanelProps = {
  title?: string;
  rows: SpecRow[];
};

export type EmptyStateProps = {
  title: string;
  description?: string;
};
