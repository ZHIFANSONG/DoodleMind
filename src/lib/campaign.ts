/**
 * 闯关：把 148 个词编成 12 关，每关 4 个词。
 *
 * 为什么要有关卡：挑战/盲区/自由是三种「规则」，但一次只玩一局，玩完就散了。
 * 关卡提供的不是新规则，而是**下一步**——你现在在第 3 关，下一个词是什么、
 * 差几颗星、下一关叫什么，这些小事比多一种玩法更能让人继续玩下去。
 *
 * 词序是按「画出来有多容易认」排的，不是随机：太阳月亮星星云是最先能画出来的
 * 东西，长颈鹿袋鼠鳄鱼留到第 10 关。每一关的四词尽量不同源，避免连续两题
 * 都是「同一个形状的变体」而有重复感。
 */

export type Stage = {
  index: number;
  /** 关卡名，显示在首页和结算页 */
  name: string;
  /** QuickDraw 英文名，必须与模型 classes 里的 `en` 完全一致 */
  words: string[];
};

export const STAGES: Stage[] = [
  { index: 0, name: '天与光', words: ['sun', 'moon', 'star', 'cloud'] },
  { index: 1, name: '院子里', words: ['cat', 'fish', 'house', 'flower'] },
  { index: 2, name: '果篮', words: ['apple', 'banana', 'strawberry', 'carrot'] },
  { index: 3, name: '随身', words: ['pencil', 'key', 'umbrella', 'clock'] },
  { index: 4, name: '餐桌', words: ['coffee cup', 'spoon', 'fork', 'knife'] },
  { index: 5, name: '上路', words: ['car', 'bus', 'train', 'bicycle'] },
  { index: 6, name: '飞与爬', words: ['bird', 'butterfly', 'bee', 'spider'] },
  { index: 7, name: '远处', words: ['tree', 'palm tree', 'mountain', 'bridge'] },
  { index: 8, name: '乐队', words: ['guitar', 'piano', 'violin', 'trumpet'] },
  { index: 9, name: '大动物', words: ['giraffe', 'elephant', 'kangaroo', 'crocodile'] },
  { index: 10, name: '天上', words: ['helicopter', 'parachute', 'hot air balloon', 'flying saucer'] },
  { index: 11, name: '终章', words: ['castle', 'windmill', 'skull', 'hourglass'] },
];

/** 每关几题（结算时用来算总星数） */
export const SLOTS_PER_STAGE = 4;
/** 每关满分 */
export const MAX_STARS_PER_STAGE = SLOTS_PER_STAGE * 3;

/**
 * 把「第几关第几题」翻译成模型里的类别下标。
 *
 * 找不到就返回 -1，让调用方退回随机出题——词表是由英文名写死的，而模型类别
 * 表来自导出时的 `words` 数组。万一哪天改了词表，宁可这题变成随机，也不要
 * 在这里抛异常把整个游戏页打黑。
 */
export function wordIndexOf(stage: number, slot: number, classes: { en: string }[]): number {
  const st = STAGES[stage];
  if (!st) return -1;
  const en = st.words[slot];
  if (!en) return -1;
  return classes.findIndex((c) => c.en === en);
}

/** 一题几颗星：越快越多，但最慢也有一颗——过关本身就值得记一笔 */
export function starsForSlot(elapsedMs: number): 1 | 2 | 3 {
  if (elapsedMs <= 9000) return 3;
  if (elapsedMs <= 20000) return 2;
  return 1;
}

/* ─────────────── 进度（纯逻辑，不碰存储） ─────────────── */

export type StageProgress = {
  /** 每题的星数，0 表示还没过。长度 = SLOTS_PER_STAGE */
  slots: number[];
  /** 这一关全部通过时的总用时（毫秒），0 表示还没通 */
  bestMs: number;
};

export type Progress = {
  stages: Record<number, StageProgress>;
};

export const EMPTY_PROGRESS: Progress = { stages: {} };

export function emptyStage(): StageProgress {
  return { slots: new Array(SLOTS_PER_STAGE).fill(0), bestMs: 0 };
}

export function stageStars(p: Progress, stage: number): number {
  const s = p.stages[stage];
  if (!s) return 0;
  return s.slots.reduce((a, b) => a + b, 0);
}

export function totalStars(p: Progress): number {
  return STAGES.reduce((a, st) => a + stageStars(p, st.index), 0);
}

/** 解锁到第几关（含）。上一关**四题都拿到过星**才开下一关。 */
export function unlockedStage(p: Progress): number {
  let i = 0;
  while (i < STAGES.length - 1) {
    const s = p.stages[i];
    if (!s || s.slots.length < SLOTS_PER_STAGE || s.slots.some((v) => v <= 0)) break;
    i += 1;
  }
  return i;
}

/** 一关里第一道还没过的题。全过了返回 -1。 */
export function nextSlot(p: Progress, stage: number): number {
  const s = p.stages[stage];
  if (!s) return 0;
  return s.slots.findIndex((v) => v <= 0);
}

/**
 * 记一题的成绩，返回新进度（纯函数，不改入参）。
 *
 * 星数只升不降：重打一题打得更差，不该把已经拿到的星改小。
 * `bestMs` 只在整关四题都过了之后才有意义——单题用时加起来不等于通关用时，
 * 中间还隔着结算、点下一题、重新出题的时间。
 */
export function applySlot(
  p: Progress,
  stage: number,
  slot: number,
  stars: number,
  elapsedMs: number,
): Progress {
  const prev = p.stages[stage] ?? emptyStage();
  const slots = prev.slots.slice();
  slots[slot] = Math.max(slots[slot] ?? 0, stars);
  const cleared = slots.every((v) => v > 0);
  const bestMs = cleared ? (prev.bestMs > 0 ? Math.min(prev.bestMs, elapsedMs) : elapsedMs) : prev.bestMs;
  return { stages: { ...p.stages, [stage]: { slots, bestMs } } };
}
