/**
 * 闯关进度的存储层。
 *
 * 和「历史记录」分开存：历史是一局一条、可以清空、可以删单条；进度是
 * 「我打到第几关了」这种不能随便丢的长期状态。混在一个 key 里，用户清一次
 * 画册就会把进度一起清掉——这两件事的危险程度完全不同。
 *
 * 所有纯逻辑（星数、解锁、下一题）都在 `campaign.ts` 里，这里只负责读写，
 * 这样它们能在 Node 里被闸门脚本直接测到（这个文件 import 了 AsyncStorage，
 * 在 Node 里跑不起来）。
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { SLOTS_PER_STAGE, STAGES, EMPTY_PROGRESS, applySlot, type Progress } from './campaign';

const KEY = 'doodlemind.progress.v1';

/**
 * 把存储里读出来的东西收进形状。
 *
 * 存坏了（手改过、旧版本结构、数组缺项）就当没存过：进度丢了是遗憾，
 * 但让游戏打不开是灾难。
 */
function normalize(raw: unknown): Progress {
  const src = raw as { stages?: Record<string, { slots?: unknown; bestMs?: unknown }> } | null;
  const stages = src?.stages;
  if (!stages) return EMPTY_PROGRESS;
  const out: Progress = { stages: {} };
  for (const k of Object.keys(stages)) {
    const i = Number(k);
    if (!Number.isInteger(i) || i < 0 || i >= STAGES.length) continue;
    const s = stages[k];
    const slots = Array.isArray(s?.slots)
      ? s.slots.map((v) => (typeof v === 'number' ? Math.max(0, Math.min(3, Math.round(v))) : 0))
      : [];
    while (slots.length < SLOTS_PER_STAGE) slots.push(0);
    out.stages[i] = {
      slots: slots.slice(0, SLOTS_PER_STAGE),
      bestMs: typeof s?.bestMs === 'number' && s.bestMs > 0 ? s.bestMs : 0,
    };
  }
  return out;
}

export async function loadProgress(): Promise<Progress> {
  try {
    const raw = await AsyncStorage.getItem(KEY);
    if (!raw) return EMPTY_PROGRESS;
    return normalize(JSON.parse(raw));
  } catch {
    return EMPTY_PROGRESS;
  }
}

async function write(p: Progress): Promise<Progress> {
  try {
    await AsyncStorage.setItem(KEY, JSON.stringify(p));
  } catch {
    // 写失败不该打断这一局
  }
  return p;
}

/**
 * 读-改-写一步到位：游戏页只需要「这一题过了」，不该自己维护一份进度副本
 * （副本一旦和存储不同步，写回去就把别处的进度覆盖掉了）。
 */
export async function saveSlot(
  stage: number,
  slot: number,
  stars: number,
  elapsedMs: number,
): Promise<Progress> {
  const p = await loadProgress();
  return write(applySlot(p, stage, slot, stars, elapsedMs));
}
