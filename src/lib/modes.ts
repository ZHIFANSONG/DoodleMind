/**
 * 三种玩法。不是换皮，是三套不同的胜负规则。
 *
 * 为什么值得做成三种而不是一种：
 * 同一个模型，把它当「要说服的裁判」和当「要骗过的对手」，玩起来完全是两个游戏。
 * 挑战模式你在配合它，反向模式你在对抗它——而对抗一个固定权重的网络，
 * 乐趣来自发现它的盲区（它认轮廓不认内部细节、它对某些朝向敏感、它被多余线条干扰）。
 * 那种「我找到这个模型的 bug 了」的瞬间，是单一模式给不了的。
 */

import type { GlyphName } from '@/components/glyph';

export type Mode = 'challenge' | 'blind' | 'free';

export type ModeSpec = {
  id: Mode;
  /** 首页大标题 */
  title: string;
  /** 一句话说清规则 */
  tagline: string;
  /** 详细规则，展开在玩法说明里 */
  rules: string[];
  durationMs: number;
  /** 是否需要目标词 */
  needsTarget: boolean;
  /** 判定通过的概率阈值 */
  passThreshold: number;
  /** 需要连续保持的毫秒数 */
  holdMs: number;
  /** 需要的最少墨量（防止空画布白嫖），单位是等效满墨像素数 */
  minInk: number;
  glyph: GlyphName;
};

export const MODES: Record<Mode, ModeSpec> = {
  challenge: {
    id: 'challenge',
    title: '挑战',
    tagline: '让它猜出你画的词',
    rules: [
      '屏幕给你一个词，你把它画出来',
      '模型的目标概率进入第一名并稳定 0.38 秒，就算它说对了',
      '60 秒内没让它说对，这局算输',
    ],
    durationMs: 60_000,
    needsTarget: true,
    passThreshold: 0.42,
    holdMs: 380,
    minInk: 0,
    glyph: 'pencil',
  },
  blind: {
    id: 'blind',
    title: '盲区',
    tagline: '让它认不出来',
    rules: [
      '屏幕给你一个词，你照样画它',
      '但只要模型把它猜成第一名，你立刻输',
      '撑满 40 秒就算你赢',
      '画得太少不算数——墨量必须够，空画布是耍赖',
      '这一局你要找的是这个模型的盲区',
    ],
    durationMs: 40_000,
    needsTarget: true,
    passThreshold: 0.42,
    holdMs: 120,
    /** 约等于「画完一个完整轮廓」的墨量。一条贯穿的粗线是 55，一只猫是 187 */
    minInk: 90,
    glyph: 'eyeSlash',
  },
  free: {
    id: 'free',
    title: '自由',
    tagline: '想画什么画什么',
    rules: [
      '没有目标词，没有倒计时',
      '候选栏实时跟着你的笔走',
      '可以当成一个「模型在怎么看你」的观察窗',
    ],
    durationMs: 0,
    needsTarget: false,
    passThreshold: 0,
    holdMs: 0,
    minInk: 0,
    glyph: 'scribble',
  },
};

export const MODE_ORDER: Mode[] = ['challenge', 'blind', 'free'];

/* ─────────────── 单局结果 ─────────────── */

export type Outcome = 'win' | 'lose' | 'draw';

export type RoundResult = {
  mode: Mode;
  outcome: Outcome;
  /** 结算卡片大标题 */
  headline: string;
  /** 一句话解释为什么 */
  detail: string;
  score: number;
  elapsedMs: number;
};

/**
 * 三种模式各自的结算。
 *
 * 分开写而不是丢一个 `cleared: boolean`，是因为三种模式的「赢」在语义上完全不同：
 * 挑战模式赢 = AI 说出了目标词；盲区模式赢 = AI 一直没说出口。
 * 用同一个布尔值表示，界面上就没法给出对的措辞，统计也会串味。
 */
export function judgeChallenge(opts: {
  cleared: boolean;
  elapsedMs: number;
  hintsUsed: number;
  streakBefore: number;
  score: number;
}): RoundResult {
  if (opts.cleared) {
    return {
      mode: 'challenge',
      outcome: 'win',
      headline: '它说出来了',
      detail: `用时 ${(opts.elapsedMs / 1000).toFixed(1)} 秒`,
      score: opts.score,
      elapsedMs: opts.elapsedMs,
    };
  }
  return {
    mode: 'challenge',
    outcome: 'lose',
    headline: '时间到',
    detail: '这次它没认出来。画大一点、线条少一点，成功率会高很多',
    score: 0,
    elapsedMs: opts.elapsedMs,
  };
}

export function judgeBlind(opts: {
  /** 模型在整局里是否曾把目标词判成第一名 */
  caught: boolean;
  elapsedMs: number;
  ink: number;
  minInk: number;
}): RoundResult {
  if (opts.caught) {
    return {
      mode: 'blind',
      outcome: 'lose',
      headline: '被它看穿了',
      detail: '它在推理途中认出了这个词。换一种画法再试',
      score: 0,
      elapsedMs: opts.elapsedMs,
    };
  }
  if (opts.ink < opts.minInk) {
    return {
      mode: 'blind',
      outcome: 'draw',
      headline: '墨太少，不算',
      detail: '得真的画点什么才行——至少把主体轮廓画完整',
      score: 0,
      elapsedMs: opts.elapsedMs,
    };
  }
  return {
    mode: 'blind',
    outcome: 'win',
    headline: '你赢了',
    detail: '整整一局它都没说出这个词。你找到了它的一处盲区',
    // 盲区模式不看速度，看「你画得多完整」：墨越多说明画得越实在，赢得越硬。
    // 300 个满墨像素（约一只画得比较满的猫）就到顶，再多不加分。
    score: Math.round(600 + Math.min(900, opts.ink * 3)),
    elapsedMs: opts.elapsedMs,
  };
}

/** 首页副标题用：把玩法压成一行 */
export function modeSummary(mode: Mode): string {
  const s = MODES[mode];
  return s.needsTarget ? `${s.durationMs / 1000} 秒 · ${s.tagline}` : s.tagline;
}
