/**
 * 一局游戏的状态机与计分。
 *
 * 核心设计：AI 是裁判，而且它对玩家是**不透明的黑箱**。
 * 玩家的任务是「把这个黑箱推到说出目标词」，这比传统的你画我猜更耐玩——
 * 你在逆向一个人的认知模型，而不是让朋友猜。
 *
 * 判定逻辑为什么不能只看 top-1：
 * 单帧 top-1 会剧烈抖动（同一张图，多画一笔排名就可能翻转）。
 * 直接要求「top-1 命中」会让游戏在手感和对错之间反复横跳，
 * 玩家体验是「明明画对了却不给过」。所以用**双条件 + 保持时长**：
 *   1. 目标类的概率 p 经 EMA 平滑后 >= 阈值
 *   2. 目标类在平滑后的排名里排第 1
 *   3. 上面两条连续保持 HOLD_MS 毫秒
 * 三条同时满足才判过。这样既稳，又不会把「差一点」的正确答案卡死。
 */

export type RoundState = 'ready' | 'playing' | 'cleared' | 'timeout' | 'skipped';

export type Guess = { index: number; p: number };

export type RoundConfig = {
  /** 一局限时（毫秒） */
  durationMs: number;
  /** 判定通过所需的平滑概率阈值 */
  passThreshold: number;
  /** 判定需要连续保持的时间 */
  holdMs: number;
  /** 开局是否直接给提示 */
  showGuesses: boolean;
};

export const DEFAULT_ROUND: RoundConfig = {
  durationMs: 60_000,
  passThreshold: 0.42,
  holdMs: 380,
  showGuesses: true,
};

/** EMA 平滑系数：越大越跟手，越小越稳 */
const EMA_ALPHA = 0.42;

export type RoundSnapshot = {
  state: RoundState;
  /** 目标类别的平滑概率 */
  closeness: number;
  /** 目标类别在平滑后的排名（0 = 第一） */
  rank: number;
  /** 已用时间 */
  elapsedMs: number;
  /** 剩余时间 */
  remainMs: number;
  /** 平滑后的前 k 名 */
  guesses: Guess[];
  /** 当前连续满足通过条件的时间 */
  holdProgress: number;
  /** 0..1，用于进度条 */
  holdRatio: number;
};

export class Round {
  readonly target: number;
  readonly config: RoundConfig;
  state: RoundState = 'ready';
  startedAt = 0;

  /** 每个类别一条 EMA 概率轨迹 */
  private ema: Float32Array;
  private holdStartedAt = 0;
  private lastInferAt = 0;
  /** 平滑后的排名 */
  private rank = 99;
  private closeness = 0;
  private guesses: Guess[] = [];
  /** 玩家用了几次提示 */
  hintsUsed = 0;

  constructor(target: number, nClasses: number, config: RoundConfig = DEFAULT_ROUND) {
    this.target = target;
    this.config = config;
    this.ema = new Float32Array(nClasses);
  }

  start(now: number): void {
    this.state = 'playing';
    this.startedAt = now;
    this.holdStartedAt = 0;
  }

  /** 喂一次推理结果。返回 true 表示这一帧刚好判定通过。 */
  feed(probs: Float32Array, now: number): boolean {
    if (this.state !== 'playing') return false;
    this.lastInferAt = now;

    // EMA 更新
    const a = EMA_ALPHA;
    for (let i = 0; i < this.ema.length; i++) {
      this.ema[i] += (probs[i] - this.ema[i]) * a;
    }

    // 排名 + 前 k
    const order = Array.from({ length: this.ema.length }, (_, i) => i);
    order.sort((x, y) => this.ema[y] - this.ema[x]);
    this.guesses = order.slice(0, 6).map((i) => ({ index: i, p: this.ema[i] }));
    this.rank = order.indexOf(this.target);
    this.closeness = this.ema[this.target];

    const pass = this.closeness >= this.config.passThreshold && this.rank === 0;
    if (pass) {
      if (this.holdStartedAt === 0) this.holdStartedAt = now;
      if (now - this.holdStartedAt >= this.config.holdMs) {
        this.state = 'cleared';
        return true;
      }
    } else {
      this.holdStartedAt = 0;
    }
    return false;
  }

  /** 每帧调用，推进超时判定 */
  tick(now: number): void {
    if (this.state !== 'playing') return;
    if (now - this.startedAt >= this.config.durationMs) {
      this.state = 'timeout';
    }
  }

  skip(): void {
    if (this.state === 'playing' || this.state === 'ready') this.state = 'skipped';
  }

  /**
   * 记一次使用提示。用方法而不是让调用方 `round.hintsUsed += 1`：
   * 后者是跨模块直接改对象字段，React Compiler 会判定为对「来自 props/state 的值」
   * 做原地变更而报错；把一个有语义的操作收进类里也更清楚。
   */
  useHint(): string {
    this.hintsUsed += 1;
    return hintFor(this.rank, this.closeness);
  }

  snapshot(now: number): RoundSnapshot {
    const elapsed = this.state === 'ready' ? 0 : now - this.startedAt;
    const holdElapsed = this.holdStartedAt === 0 ? 0 : now - this.holdStartedAt;
    return {
      state: this.state,
      closeness: this.closeness,
      rank: this.rank,
      elapsedMs: elapsed,
      remainMs: Math.max(0, this.config.durationMs - elapsed),
      guesses: this.guesses,
      holdProgress: holdElapsed,
      holdRatio: Math.min(1, holdElapsed / this.config.holdMs),
    };
  }
}

/**
 * 得分：基础 1000，每秒扣 12，用一次提示扣 120，连击每级加 15%。
 * 最低保底 100，避免扫兴。
 */
export function scoreRound(opts: {
  elapsedMs: number;
  hintsUsed: number;
  streakBefore: number;
}): number {
  const base = 1000 - (opts.elapsedMs / 1000) * 12 - opts.hintsUsed * 120;
  const streakBonus = 1 + Math.min(5, opts.streakBefore) * 0.15;
  return Math.max(100, Math.round(base * streakBonus));
}

/**
 * 提示：按当前排名把目标词的「方向」说出来。
 * 不给答案，只给模型的相对判断——保持游戏性，同时避免玩家卡死。
 */
export function hintFor(rank: number, closeness: number): string {
  if (closeness >= 0.3 || rank === 0) return '它快说出来了，再加两笔特征';
  if (rank <= 2) return '很接近，它已经把它放进前三了';
  if (rank <= 5) return '有点意思，模型开始往这个方向想了';
  if (rank <= 15) return '方向偏了，试试换个角度画（它认的是轮廓，不是细节）';
  return '完全没认出来，画大一点、线条少一点、别画太挤';
}
