/**
 * 一局的状态机。**按模式分派，不是一个通用的"对/错"开关。**
 *
 * 挑战模式的赢 = 模型说出目标词；
 * 盲区模式的赢 = 模型整局都没说出目标词。
 * 两者的判定条件几乎完全相反，硬塞进一个 `cleared: boolean` 只会让
 * 结算文案、统计、存档全部串味。所以这里让 Session 自己知道自己在judge什么。
 *
 * 判定为什么不能只看单帧 top-1：
 * 同一张图多画一笔，排名就可能翻转。直接要求「这一帧 top-1 命中」会让玩家
 * 体验在「明明画对了却不给过」和「随便画两笔就过了」之间反复横跳。
 * 所以用 **EMA 平滑 + 阈值 + 连续保持时长** 三条同时成立。
 */

import { EmaTracker, hintFor, scoreRound, type Guess } from './game';
import {
  MODES,
  judgeBlind,
  judgeChallenge,
  type Mode,
  type RoundResult,
} from './modes';

export type SessionState = 'ready' | 'playing' | 'won' | 'lost' | 'draw';

export type Snapshot = {
  state: SessionState;
  /** 目标类别的平滑概率（自由模式为 0） */
  closeness: number;
  /** 目标类别在平滑后的排名，0 = 第一（自由模式为 -1） */
  rank: number;
  elapsedMs: number;
  remainMs: number;
  guesses: Guess[];
  /** 0..1，判定进度条 */
  holdRatio: number;
  /** 盲区模式：模型已经看穿了吗（用来做紧张感提示） */
  danger: boolean;
};

export class Session {
  readonly mode: Mode;
  readonly target: number;
  readonly nClasses: number;
  readonly tracker: EmaTracker;
  /**
   * 进这一局之前已经连赢了几局。
   *
   * 由调用方传进来而不是让 Session 自己记：Session 的寿命只有一局，
   * 一局结束时它就死了，跨局的东西不该住在里面。放外面还有个好处——
   * 连击是玩家看得见的状态，存在页面上更容易做动画。
   */
  readonly streakBefore: number;

  state: SessionState = 'ready';
  startedAt = 0;
  hintsUsed = 0;

  /** 盲区模式：模型是否曾经看穿 */
  private caught = false;
  /**
   * 「达标从这一刻开始算」的时间戳。-1 表示还没开始保持。
   *
   * 不能用 0 当哨兵：0 是合法时间戳（测试里就常用 now=0 起算），
   * 一旦真的传进来一个 0，第一帧把 holdStartedAt 写成 0 之后，
   * 下一帧又会判成「还没开始」→ 保持时长永远攒不满，判定静默失效。
   */
  private holdStartedAt = -1;
  /** 最近一次喂进来的墨量，结算时用 */
  ink = 0;

  constructor(mode: Mode, target: number, nClasses: number, streakBefore = 0) {
    this.mode = mode;
    this.target = target;
    this.nClasses = nClasses;
    this.streakBefore = streakBefore;
    this.tracker = new EmaTracker(nClasses);
    // 自由模式没有目标、没有时限，进页面就算「进行中」
    if (mode === 'free') this.state = 'playing';
  }

  get spec() {
    return MODES[this.mode];
  }

  start(now: number): void {
    if (this.state !== 'ready') return;
    this.state = 'playing';
    this.startedAt = now;
  }

  /**
   * 喂一次推理结果。
   * @returns 本帧是否产生了终局（赢了或输了），无终局返回 null
   */
  feed(probs: Float32Array, now: number, ink: number): RoundResult | null {
    if (this.state !== 'playing') return null;
    this.ink = ink;
    this.tracker.update(probs);

    if (this.mode === 'free') return null;

    const spec = this.spec;
    const p = this.tracker.pOf(this.target);
    const rank = this.tracker.rankOf(this.target);
    const hit = rank <= spec.passRank && p >= spec.passThreshold;

    if (this.mode === 'challenge') {
      if (hit) {
        if (this.holdStartedAt < 0) this.holdStartedAt = now;
        if (now - this.holdStartedAt >= spec.holdMs) {
          this.state = 'won';
          return judgeChallenge({
            cleared: true,
            elapsedMs: now - this.startedAt,
            hintsUsed: this.hintsUsed,
            streakBefore: this.streakBefore,
            score: scoreRound({
              elapsedMs: now - this.startedAt,
              hintsUsed: this.hintsUsed,
              streakBefore: this.streakBefore,
            }),
          });
        }
      } else {
        this.holdStartedAt = -1;
      }
      return null;
    }

    // 盲区模式：被看穿就立刻输。保持时长设得很短（120ms），
    // 因为这里的"看穿"是个坏消息，不应该给玩家 0.38 秒的侥幸窗口。
    if (hit) {
      if (this.holdStartedAt < 0) this.holdStartedAt = now;
      if (now - this.holdStartedAt >= spec.holdMs) {
        this.caught = true;
        this.state = 'lost';
        return judgeBlind({
          caught: true,
          elapsedMs: now - this.startedAt,
          ink: this.ink,
          minInk: spec.minInk,
        });
      }
    } else {
      this.holdStartedAt = -1;
    }
    return null;
  }

  /** 每帧推进超时。挑战模式超时算输，盲区模式超时算赢。 */
  tick(now: number): RoundResult | null {
    if (this.state !== 'playing') return null;
    const spec = this.spec;
    if (spec.durationMs === 0) return null;
    if (now - this.startedAt < spec.durationMs) return null;

    const elapsed = now - this.startedAt;
    if (this.mode === 'challenge') {
      this.state = 'lost';
      return judgeChallenge({
        cleared: false,
        elapsedMs: elapsed,
        hintsUsed: this.hintsUsed,
        streakBefore: this.streakBefore,
        score: 0,
      });
    }
    this.state = this.ink < spec.minInk ? 'draw' : 'won';
    return judgeBlind({
      caught: false,
      elapsedMs: elapsed,
      ink: this.ink,
      minInk: spec.minInk,
    });
  }

  /** 玩家主动放弃（挑战模式跳过 / 盲区模式认输） */
  giveUp(now: number): RoundResult | null {
    if (this.state !== 'playing') return null;
    const elapsed = now - this.startedAt;
    if (this.mode === 'challenge') {
      this.state = 'lost';
      return judgeChallenge({
        cleared: false,
        elapsedMs: elapsed,
        hintsUsed: this.hintsUsed,
        streakBefore: this.streakBefore,
        score: 0,
      });
    }
    if (this.mode === 'blind') {
      this.state = 'lost';
      return judgeBlind({ caught: true, elapsedMs: elapsed, ink: this.ink, minInk: this.spec.minInk });
    }
    return null;
  }

  /**
   * 记一次提示。
   * 用方法而不是让调用方 `s.hintsUsed += 1`：后者是跨模块原地改字段，
   * React Compiler 会判定成对 props/state 的变更而报错。
   */
  useHint(): string {
    this.hintsUsed += 1;
    return hintFor(this.tracker.rankOf(this.target), this.tracker.pOf(this.target));
  }

  snapshot(now: number): Snapshot {
    const spec = this.spec;
    // 自由模式在构造函数里就是 'playing'，但从未调 start()，startedAt 还是 0。
    // 如果不排除它，elapsed 会等于「1970 年到现在」的毫秒数——页面上直接显示
    // 1794912810s 这种数字。写在这里而不是页面里，是因为任何未来取 elapsed 的
    // 地方都会踩同一个坑。
    const elapsed =
      this.mode === 'free' || this.state === 'ready' || this.startedAt === 0
        ? 0
        : Math.max(0, now - this.startedAt);
    const holdElapsed = this.holdStartedAt < 0 ? 0 : now - this.holdStartedAt;
    const total = spec.durationMs || 1;
    const rank = this.mode === 'free' ? -1 : this.tracker.rankOf(this.target);
    const closeness = this.mode === 'free' ? 0 : this.tracker.pOf(this.target);

    return {
      state: this.state,
      closeness,
      rank,
      elapsedMs: elapsed,
      // 盲区模式倒计时是「还能撑多久」，所以也是 total - elapsed
      remainMs: spec.durationMs === 0 ? 0 : Math.max(0, total - elapsed),
      guesses: this.tracker.guesses,
      holdRatio:
        spec.durationMs === 0 || spec.holdMs === 0
          ? 0
          : Math.min(1, holdElapsed / spec.holdMs),
      // 盲区模式的紧张感来源：模型已经进前三但还没说出口
      danger: this.mode === 'blind' && rank >= 0 && rank <= 2 && closeness > 0.12,
    };
  }
}
