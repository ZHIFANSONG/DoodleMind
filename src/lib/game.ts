/**
 * 候选栏的平滑与排序，加上计分和提示文案。
 *
 * 这一层和「一局的状态机」（`session.ts`）分开：平滑排序是所有玩法共用的，
 * 而胜负规则每种玩法都不一样。混在一起的话，加一个新模式就得动判定逻辑，
 * 迟早会写出「改了挑战模式，盲区模式跟着变了」这种耦合。
 */

export type Guess = { index: number; p: number };

/** 候选栏显示几个 */
export const TOP_K = 6;

/**
 * EMA 平滑系数。越大越跟手，越小越稳。
 *
 * 0.42 是调出来的：再大，手指抖动会直接反映到候选栏，词条疯狂跳；
 * 再小，画完一笔要等半秒才看到变化，像卡了。
 */
const EMA_ALPHA = 0.42;

export class EmaTracker {
  readonly n: number;
  private ema: Float32Array;
  private order: number[];
  private _guesses: Guess[] = [];

  constructor(n: number) {
    this.n = n;
    this.ema = new Float32Array(n);
    this.order = Array.from({ length: n }, (_, i) => i);
  }

  update(probs: Float32Array): void {
    const a = EMA_ALPHA;
    for (let i = 0; i < this.n; i++) {
      this.ema[i] += (probs[i] - this.ema[i]) * a;
    }
    // 每帧全量排序 148 个 float。比维护部分有序表简单得多，
    // 而且 148 的常数小到无所谓（实测 < 0.02ms）。
    this.order.sort((x, y) => this.ema[y] - this.ema[x]);
    this._guesses = this.order.slice(0, TOP_K).map((i) => ({ index: i, p: this.ema[i] }));
  }

  get guesses(): Guess[] {
    return this._guesses;
  }

  rankOf(i: number): number {
    return this.order.indexOf(i);
  }

  pOf(i: number): number {
    return i < 0 ? 0 : (this.ema[i] ?? 0);
  }

  get top(): Guess | undefined {
    return this._guesses[0];
  }

  reset(): void {
    this.ema.fill(0);
    this._guesses = [];
  }
}

/**
 * 得分：基础 1000，每秒扣 12，用一次提示扣 120，连击每级加 15%。
 * 最低保底 100，避免好不容易猜中还给 0 分。
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
 * 提示：按当前排名把目标的「方向」说出来，不给答案。
 *
 * 刻意分段而不是给一个百分比——玩家需要的是「我该往哪改」，
 * 而 0.13 这种数字只会让人更迷茫。
 */
export function hintFor(rank: number, closeness: number): string {
  if (closeness >= 0.3 || rank === 0) return '它快说出来了，再加两笔特征';
  if (rank <= 2) return '很接近，它已经把它放进前三了';
  if (rank <= 5) return '有点意思，模型开始往这个方向想了';
  if (rank <= 15) return '方向偏了，换个角度画（它认轮廓，不认细节）';
  return '完全没认出来。画大一点、线条少一点、别画太挤';
}
