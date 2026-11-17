/**
 * DoodleMind 设计系统。
 *
 * 整体走「纸与墨」：暖白纸面 + 近黑墨线 + 一抹朱红印章色。
 * 不用深色科技风——你画我猜的载体是手绘，暖白纸面更贴合笔触，
 * 也让墨迹在屏幕上看起来真的像画的，而不是发光的像素。
 */

import { Platform } from 'react-native';

export const Palette = {
  /** 纸面：微微发暖的白 */
  paper: '#FBFAF6',
  /** 卡片 / 抬升面 */
  card: '#FFFFFF',
  /** 次级纸面（画布区略暗一点，让墨迹更清楚） */
  paperSunken: '#F4F2EC',
  /** 主墨色：文字、笔画 */
  ink: '#17161A',
  /** 次级墨色 */
  inkSoft: '#5B5A63',
  /** 更淡：辅助信息 */
  inkFaint: '#9C9BA4',
  /** 朱红：印章、当前题目、猜中高亮 */
  seal: '#D9422B',
  /** 朱红浅调：背景块 */
  sealWash: '#FBEAE6',
  /** 靛蓝：次要强调、进度 */
  indigo: '#2F5BD4',
  indigoWash: '#E8EDFC',
  /** 竹青：成功 / 已猜出 */
  jade: '#1F8A5B',
  jadeWash: '#E4F3EC',
  /** 分隔线 */
  rule: '#E6E3DA',
} as const;

export const Radius = {
  sm: 8,
  md: 14,
  lg: 22,
  pill: 999,
} as const;

export const Space = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 18,
  xl: 26,
  xxl: 38,
} as const;

export const Typography = {
  /** 大标题：题目名 */
  title: Platform.select({ ios: 'Georgia-Bold', default: 'serif' }),
  /** 正文 */
  body: Platform.select({ ios: 'System', default: 'normal' }),
  /** 数字（计时、置信度）——等宽防跳动 */
  mono: Platform.select({ ios: 'Courier', default: 'monospace' }),
  /** 中文标题用系统字体更稳（Georgia 不含中文字形） */
  zh: Platform.select({ ios: 'PingFangSC-Semibold', default: 'normal' }),
} as const;

/** 笔刷色板：给玩家选色 */
export const BrushColors = [
  '#17161A',
  '#D9422B',
  '#E8912A',
  '#2F5BD4',
  '#1F8A5B',
  '#7B4BD6',
] as const;
