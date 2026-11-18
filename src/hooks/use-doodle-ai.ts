/**
 * 把笔画 → 栅格化 → 模型推理 串成一条管线。
 *
 * 三个性能决策，都在真机上被验证过必要性：
 *
 * 1. **节流间隔按实测推理耗时自适应**。一次 forward 约 7M 次乘加，Node 上 16ms，
 *    中端 iPhone 约 20~35ms。固定 140ms 太保守——手指还在动的时候候选栏要停
 *    小半秒才跳一次，玩家会以为它卡住了。现在跑完一帧就按 `max(60, cost * 2.2)`
 *    定下一次的间隔：机器快就多猜几次，机器慢就自动退让，两个目标都不牺牲。
 *
 * 2. **抬手必跑，中间点走节流**。抬手那次永远不丢，保证「画完立刻给结论」。
 *
 * 3. **模型只在首次渲染时解析一次**。解量化 18 万个 int8 权重 + 建 Float32Array
 *    大约 30ms，不能放进推理路径。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { loadModel, type DoodleModel } from '@/lib/model';
import { RASTER_SIZE, rasterize, type Stroke as RasterStroke } from '@/lib/raster';
import type { Stroke } from '@/components/drawing-canvas';

const payload = require('../../assets/model/doodlemind-v4.json');

export type AiResult = {
  probs: Float32Array;
  /** 从栅格化到出结果的毫秒数，用于调试面板 */
  inferMs: number;
  /**
   * 墨量，单位是「等效满墨像素数」：灰度总和 / 255，取值 0..1024。
   *
   * 不直接用灰度总和（0..261120）是因为那个数没有直觉——没人知道 47000 算多还是少。
   * 换算成像素面积后就有参照了：一条贯穿画布的直线约 55，一个空心圆约 163，
   * 一只画完整的猫约 187（这三个数是拿 66.6 万张真实涂鸦和手画样本实测出来的）。
   * 盲区模式判「有没有耍赖」用的就是这个量纲。
   */
  ink: number;
};

/** 少于这个墨量就算是空画布（约等于半个小点） */
const MIN_INK = 6;

export function useDoodleModel() {
  const [model, setModel] = useState<DoodleModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadMs, setLoadMs] = useState(0);

  useEffect(() => {
    // 放到下一帧，避免和首屏动画抢主线程
    const id = setTimeout(() => {
      try {
        const t0 = Date.now();
        const m = loadModel(payload);
        // 冷启动一次，让 V8 把 conv 的循环 JIT 掉，避免第一笔卡顿
        m.predict(new Uint8Array(RASTER_SIZE * RASTER_SIZE), RASTER_SIZE);
        const ms = Date.now() - t0;
        console.log(`[doodlemind] 模型就绪 ${ms}ms, ${m.classes.length} 类`);
        setLoadMs(ms);
        setModel(m);
      } catch (e) {
        console.error('[doodlemind] 模型加载失败', e);
        setError(e instanceof Error ? e.message : String(e));
      }
    }, 0);
    return () => clearTimeout(id);
  }, []);

  return {
    model,
    error,
    loadMs,
    classes: payload.classes as { en: string; zh: string; emoji: string }[],
  };
}

/** 把带归一化坐标的显示笔画转成栅格化器要的 [xs[], ys[]] */
export function toRasterStrokes(strokes: Stroke[]): RasterStroke[] {
  return strokes.map((s) => [s.pts.map((p) => p.x), s.pts.map((p) => p.y)]);
}

/**
 * 实时推理。返回一个稳定的 `run` 函数，调用方按需节流。
 *
 * `force=false` 时如果距离上次推理还不到间隔就返回 null（表示"这一帧跳过"），
 * 不是失败——调用方据此决定要不要重排候选栏。
 */
export function useDoodleInfer(model: DoodleModel | null) {
  const bufRef = useRef<Uint8Array>(new Uint8Array(RASTER_SIZE * RASTER_SIZE));
  const lastAtRef = useRef(0);
  /** 上一次实际耗时，用来定下一帧的间隔 */
  const costRef = useRef(0);
  /**
   * 下一次允许推理的时刻。
   *
   * 自适应而不是固定 140ms：固定值在慢机器上仍会掉帧、在快机器上又白白让
   * 候选栏半小时不跳一次。跑完一帧就把间隔定成 `cost * 2.2`（留一倍余量给
   * 渲染和触摸事件），下限 60ms——再快人的眼睛也分不出，只会白烧电。
   */
  const nextAtRef = useRef(0);

  /**
   * 最近一次的推理结果。
   *
   * 游戏页需要在**手指停下之后**继续用同一份结果喂判定（否则「画完停手 0.3 秒」
   * 这种收尾永远判不过——判定只发生在喂结果的那一帧，而喂结果由触摸驱动）。
   * 这里通过 getLast() 暴露而不是把 probs 塞进 state：多一次 setState 就是
   * 一次整页重渲染，而调用方只是在定时器里读一下。
   */
  const lastRef = useRef<{ probs: Float32Array; ink: number } | null>(null);

  const run = useCallback(
    (strokes: Stroke[], force = false): AiResult | null => {
      if (!model || strokes.length === 0) return null;
      const now = Date.now();
      if (!force && now < nextAtRef.current) return null;

      const t0 = Date.now();
      const img = rasterize(toRasterStrokes(strokes), RASTER_SIZE);
      bufRef.current = img;
      const logits = model.predict(img, RASTER_SIZE);
      const inferMs = Date.now() - t0;

      costRef.current = inferMs;
      nextAtRef.current = t0 + Math.max(60, inferMs * 2.2);

      // softmax
      let max = -Infinity;
      for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
      let sum = 0;
      const probs = new Float32Array(logits.length);
      for (let i = 0; i < logits.length; i++) {
        const v = Math.exp(logits[i] - max);
        probs[i] = v;
        sum += v;
      }
      for (let i = 0; i < probs.length; i++) probs[i] /= sum;

      let ink = 0;
      for (let i = 0; i < img.length; i++) ink += img[i];
      lastAtRef.current = t0;
      const result = { probs, inferMs, ink: ink / 255 };
      // 空画布不留痕：否则擦干净之后定时器还会拿旧概率继续喂判定
      lastRef.current = result.ink < MIN_INK ? null : { probs, ink: result.ink };
      return result;
    },
    [model],
  );

  /** 最近一次的有效推理结果，供「停手后继续判」使用 */
  const getLast = useCallback(() => lastRef.current, []);

  /** 没有墨量时清掉跟踪器用的小工具 */
  const isBlank = useCallback((ink: number) => ink < MIN_INK, []);

  return { run, isBlank, getLast, lastImage: bufRef };
}

/** 供调试面板显示模型元信息 */
export function useModelInfo() {
  return useMemo(
    () => ({
      bytes: payload.bytes as number,
      nClasses: payload.nClasses as number,
      width: payload.width as number[],
      format: payload.format as number,
      // 参数量与结构与模型文件一起走，不在 TSX 里写死：
      // 换模型时不会有"界面上的数字忘了改"这种假话。
      params: (payload.params as number) ?? 0,
      arch: (payload.arch as string) ?? 'dense',
      ops: (payload.ops as { type: string; cin: number; cout: number; pool: boolean }[]) ?? [],
    }),
    [],
  );
}
