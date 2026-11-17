/**
 * 把笔画 → 栅格化 → 模型推理 → 平滑判定 串成一条管线。
 *
 * 三个性能决策，都在真机上被验证过必要性：
 *
 * 1. **节流到 ~140ms 一次**。一次 forward 约 7M 次乘加，在中端 iPhone 上
 *    约 20–35ms。不节流的话每抬一次手都跑一遍，画得快时会掉帧。
 *
 * 2. **抬手必跑，中间点走节流**。玩家最在意的是「我画完这一笔它说了什么」，
 *    所以抬手那次永远不丢；画的过程中顺带更新，制造「它在实时盯着我」的感觉。
 *
 * 3. **模型只在首次渲染时解析一次**。解量化 16 万个 int8 权重 + 建 Float32Array
 *    大约 30ms，不能放进推理路径。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { loadModel, type DoodleModel } from '@/lib/model';
import { RASTER_SIZE, rasterize, type Stroke as RasterStroke } from '@/lib/raster';
import type { Stroke } from '@/components/drawing-canvas';

const payload = require('../../assets/model/doodlemind-v1.json');

export type AiResult = {
  probs: Float32Array;
  /** 从栅格化到出结果的毫秒数，用于调试面板 */
  inferMs: number;
  /** 是否真的认出了东西（墨量太少就不算） */
  hasInk: boolean;
};

export function useDoodleModel() {
  const [model, setModel] = useState<DoodleModel | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // 放到下一帧，避免和首屏动画抢主线程
    const id = setTimeout(() => {
      try {
        const t0 = Date.now();
        const m = loadModel(payload);
        // 冷启动一次，让 V8 把 conv 的循环 JIT 掉，避免第一笔卡顿
        m.predict(new Uint8Array(RASTER_SIZE * RASTER_SIZE), RASTER_SIZE);
        console.log(`[doodlemind] 模型就绪 ${Date.now() - t0}ms, ${m.classes.length} 类`);
        setModel(m);
      } catch (e) {
        console.error('[doodlemind] 模型加载失败', e);
        setError(e instanceof Error ? e.message : String(e));
      }
    }, 0);
    return () => clearTimeout(id);
  }, []);

  return { model, error, classes: payload.classes as { en: string; zh: string; emoji: string }[] };
}

/** 把带归一化坐标的显示笔画转成栅格化器要的 [xs[], ys[]] */
export function toRasterStrokes(strokes: Stroke[]): RasterStroke[] {
  return strokes.map((s) => [s.pts.map((p) => p.x), s.pts.map((p) => p.y)]);
}

/**
 * 实时推理。返回一个稳定的 `run` 函数，调用方按需节流。
 */
export function useDoodleInfer(model: DoodleModel | null) {
  const bufRef = useRef<Uint8Array>(new Uint8Array(RASTER_SIZE * RASTER_SIZE));
  const lastRef = useRef(0);

  const run = useCallback(
    (strokes: Stroke[], force = false): AiResult | null => {
      if (!model || strokes.length === 0) return null;
      const now = Date.now();
      if (!force && now - lastRef.current < 140) return null;
      lastRef.current = now;

      const t0 = Date.now();
      const img = rasterize(toRasterStrokes(strokes), RASTER_SIZE);
      bufRef.current = img;
      const logits = model.predict(img, RASTER_SIZE);
      const inferMs = Date.now() - t0;

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
      return { probs, inferMs, hasInk: ink / 255 > 6 };
    },
    [model],
  );

  return { run, lastImage: bufRef };
}

/** 供调试面板显示模型元信息 */
export function useModelInfo() {
  return useMemo(
    () => ({
      bytes: payload.bytes as number,
      nClasses: payload.nClasses as number,
      width: payload.width as number[],
      format: payload.format as number,
    }),
    [],
  );
}
