/**
 * 历史记录存储。
 *
 * 只存**矢量笔画**，不存截图：一张 1080p PNG 约 400KB，一局 20 条就 8MB；
 * 同样一局的笔画 JSON 只有 8–20KB，缩略图还能按需重绘成任意分辨率。
 * 这是「存结构不存像素」的典型收益。
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import type { Pt, Stroke } from '@/components/drawing-canvas';

/**
 * 一条对局记录。
 *
 * 名字故意不叫 `Record`：那会遮蔽 TS 内置的 `Record<K,V>` 工具类型，
 * 以后在这个文件里写 `Record<string, number>` 会莫名其妙报错。
 */
export type GameRecord = {
  id: string;
  /** 目标类别下标 */
  target: number;
  /** 目标词（存一份冗余，防止以后类别表变动导致历史错乱） */
  targetZh: string;
  targetEn: string;
  emoji: string;
  /** 是否在时限内让 AI 说出来 */
  cleared: boolean;
  /** 用时（毫秒） */
  elapsedMs: number;
  score: number;
  /** 笔画数 */
  strokeCount: number;
  /** 落盘时间戳 */
  at: number;
  strokes: { pts: [number, number][]; color: string; width: number }[];
};

const KEY = 'doodlemind.history.v1';
const MAX_KEEP = 200;

/** 序列化时把 {x,y} 压成 [x,y] —— JSON 体积能小 40% */
function packStrokes(strokes: Stroke[]): GameRecord['strokes'] {
  return strokes.map((s) => ({
    pts: s.pts.map((p) => [round(p.x), round(p.y)] as [number, number]),
    color: s.color,
    width: round(s.width),
  }));
}

function round(v: number): number {
  return Math.round(v * 10000) / 10000;
}

export function unpackStrokes(raw: GameRecord['strokes']): Stroke[] {
  return raw.map((s) => ({
    pts: s.pts.map(([x, y]) => ({ x, y }) as Pt),
    color: s.color,
    width: s.width,
  }));
}

export async function loadHistory(): Promise<GameRecord[]> {
  try {
    const raw = await AsyncStorage.getItem(KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw) as GameRecord[];
    return Array.isArray(arr) ? arr : [];
  } catch (e) {
    console.warn('[doodlemind] 读历史失败', e);
    return [];
  }
}

export async function appendRecord(
  r: Omit<GameRecord, 'id' | 'at' | 'strokes'> & { strokes: Stroke[] },
): Promise<GameRecord[]> {
  const list = await loadHistory();
  const item: GameRecord = {
    ...r,
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    at: Date.now(),
    strokes: packStrokes(r.strokes),
  };
  const next = [item, ...list].slice(0, MAX_KEEP);
  try {
    await AsyncStorage.setItem(KEY, JSON.stringify(next));
  } catch (e) {
    console.warn('[doodlemind] 写历史失败', e);
  }
  return next;
}

export async function clearHistory(): Promise<void> {
  await AsyncStorage.removeItem(KEY);
}

export async function removeRecord(id: string): Promise<GameRecord[]> {
  const list = await loadHistory();
  const next = list.filter((r) => r.id !== id);
  await AsyncStorage.setItem(KEY, JSON.stringify(next));
  return next;
}

/** 汇总统计：总局数、通过数、最高分、总笔画数 */
export function summarize(list: GameRecord[]) {
  const cleared = list.filter((r) => r.cleared).length;
  const best = list.reduce((m, r) => Math.max(m, r.score), 0);
  const totalStrokes = list.reduce((s, r) => s + r.strokeCount, 0);
  const avgMs = cleared
    ? Math.round(list.filter((r) => r.cleared).reduce((s, r) => s + r.elapsedMs, 0) / cleared)
    : 0;
  return { rounds: list.length, cleared, best, totalStrokes, avgMs };
}

