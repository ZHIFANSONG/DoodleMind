/**
 * 端上 CNN 推理引擎。纯 TypeScript，无任何原生依赖、无 tfjs。
 *
 * 网络结构（与 `tools/train.py::DoodleNet` 严格一致）：
 *   输入 1x32x32
 *   block1: conv3x3(1→24)  + ReLU   @32, maxpool→16
 *   block2: conv3x3(24→48) + ReLU   @16, maxpool→8
 *   block3: conv3x3(48→96) + ReLU   @8,  maxpool→4
 *   block4: conv3x3(96→128)+ ReLU   @4
 *   GAP → FC(128→N) → softmax
 *
 * 权重以 per-output-channel 对称 int8 存储，首次加载时一次性解量化成
 * Float32Array 常驻内存（约 650KB）。推理全程 float32 —— JS 里没有 SIMD，
 * 整数运算并不比浮点快，而激活量化会累积误差，不划算。
 *
 * 为什么不用 tfjs：tfjs 的 wasm 后端在 Expo Go 里跑不了（要 WASM 支持），
 * 纯 JS 后端又比这份手写代码慢 3-5 倍，还要多 2MB 依赖。
 */

export type ModelClass = { en: string; zh: string; emoji: string };

export type DoodleModel = {
  classes: ModelClass[];
  /** 跑一次 forward，返回各类 logits */
  predict(img: Uint8Array, size: number): Float32Array;
  /** top-k 结果，按概率降序 */
  topK(img: Uint8Array, size: number, k: number): { index: number; p: number }[];
};

type Layer = {
  cin: number;
  cout: number;
  /** (cout * cin * 9) 解量化后的 float32，行主序 [co][ci][ky][kx] */
  w: Float32Array;
  bias: Float32Array;
};

const MAGIC = 0x4e4d4444; // "DDMN" 小端读成 u32

function decodeBase64(b64: string): Uint8Array {
  // RN 没有 atob 的保证，手写一份（同时避免 Hermes 的 Buffer 缺失）
  const map =
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  const lut = new Int16Array(256).fill(-1);
  for (let i = 0; i < map.length; i++) lut[map.charCodeAt(i)] = i;

  let pad = 0;
  if (b64.endsWith('==')) pad = 2;
  else if (b64.endsWith('=')) pad = 1;

  const len = b64.length;
  const out = new Uint8Array(((len / 4) | 0) * 3 - pad);
  let o = 0;
  for (let i = 0; i < len; i += 4) {
    const c0 = lut[b64.charCodeAt(i)];
    const c1 = lut[b64.charCodeAt(i + 1)];
    const c2 = lut[b64.charCodeAt(i + 2)];
    const c3 = lut[b64.charCodeAt(i + 3)];
    const triple = (c0 << 18) | (c1 << 12) | ((c2 < 0 ? 0 : c2) << 6) | (c3 < 0 ? 0 : c3);
    if (o < out.length) out[o++] = (triple >> 16) & 0xff;
    if (o < out.length) out[o++] = (triple >> 8) & 0xff;
    if (o < out.length) out[o++] = triple & 0xff;
  }
  return out;
}

/** 从二进制 blob 构建模型 */
export function parseModel(bytes: Uint8Array, classes: ModelClass[]): DoodleModel {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let p = 0;

  const magic = view.getUint32(0, true);
  if (magic !== MAGIC) throw new Error(`模型文件 magic 不对: 0x${magic.toString(16)}`);
  p = 4;
  const version = view.getUint8(p++);
  if (version !== 2) throw new Error(`模型版本 ${version} 不支持（期望 2）`);
  const nClasses = view.getUint16(p, true);
  p += 2;
  const nBlocks = view.getUint16(p, true);
  p += 2;
  if (nClasses !== classes.length) {
    throw new Error(`类别数不一致：模型 ${nClasses} vs 词表 ${classes.length}`);
  }

  const layers: Layer[] = [];
  for (let bi = 0; bi < nBlocks; bi++) {
    const cin = view.getUint16(p, true);
    p += 2;
    const cout = view.getUint16(p, true);
    p += 2;

    const wScale = new Float32Array(cout);
    for (let i = 0; i < cout; i++) {
      wScale[i] = view.getFloat32(p, true);
      p += 4;
    }
    const bias = new Float32Array(cout);
    for (let i = 0; i < cout; i++) {
      bias[i] = view.getFloat32(p, true);
      p += 4;
    }

    const n = cout * cin * 9;
    const w = new Float32Array(n);
    // 解量化：q * wScale[co]
    let q = p;
    let oi = 0;
    for (let co = 0; co < cout; co++) {
      const s = wScale[co];
      for (let t = 0; t < cin * 9; t++) {
        w[oi++] = view.getInt8(q) * s;
        q += 1;
      }
    }
    p += n;
    layers.push({ cin, cout, w, bias });
  }

  const width = view.getUint16(p, true);
  p += 2;
  const headW = new Float32Array(nClasses * width);
  for (let i = 0; i < headW.length; i++) {
    headW[i] = view.getFloat32(p, true);
    p += 4;
  }
  const headB = new Float32Array(nClasses);
  for (let i = 0; i < nClasses; i++) {
    headB[i] = view.getFloat32(p, true);
    p += 4;
  }

  // 复用的中间缓冲。最大尺寸 = 第一层 32x32x24，之后越来越小，
  // 统一按 32*32*maxCout 分配一次，避免每帧 GC。
  const maxCout = Math.max(...layers.map((l) => l.cout));
  let bufA = new Float32Array(32 * 32 * maxCout);
  let bufB = new Float32Array(32 * 32 * maxCout);

  /** 3x3 pad=1 conv + ReLU。in: (cin,h,w) 行主序 */
  function conv3x3(
    input: Float32Array,
    layer: Layer,
    h: number,
    wd: number,
    out: Float32Array,
  ): void {
    const { cin, cout, w, bias } = layer;
    const area = h * wd;
    out.fill(0, 0, cout * area);
    // 先把 bias 铺进去
    for (let co = 0; co < cout; co++) {
      const b = bias[co];
      const base = co * area;
      out.fill(b, base, base + area);
    }

    for (let ci = 0; ci < cin; ci++) {
      const inBase = ci * area;
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const din = ky - 1;
          const dkx = kx - 1;
          // 输出 (y,x) 读输入 (y+din, x+dkx)。有效条件是 0 <= y+din <= h-1，
          // 即 y ∈ [max(0,-din), min(h, h-din))。
          //
          // 这里极易写反：din>0（向下偏移）时上界是 h-din，din<0 才是 h。
          // 写反的后果不是崩溃，而是读到相邻通道的内存——数组本身不会越界
          // （后面还有别的通道），模型照样出结果，只是悄悄变笨。
          // 当年就是靠跨语言一致率从 100% 掉到 57% 才把它抓出来。
          const y0 = din < 0 ? -din : 0;
          const y1 = din > 0 ? h - din : h;
          const x0 = dkx < 0 ? -dkx : 0;
          const x1 = dkx > 0 ? wd - dkx : wd;
          if (y1 <= y0 || x1 <= x0) continue;
          for (let co = 0; co < cout; co++) {
            const k = w[(co * cin + ci) * 9 + ky * 3 + kx];
            if (k === 0) continue;
            const outBase = co * area;
            for (let y = y0; y < y1; y++) {
              const inRow = inBase + (y + din) * wd;
              const outRow = outBase + y * wd;
              for (let x = x0; x < x1; x++) {
                out[outRow + x] += k * input[inRow + x + dkx];
              }
            }
          }
        }
      }
    }

    // ReLU
    const total = cout * area;
    for (let i = 0; i < total; i++) {
      if (out[i] < 0) out[i] = 0;
    }
  }

  /** 2x2 stride2 maxpool。in: (c,h,w) → out: (c,h/2,w/2) */
  function maxpool2(input: Float32Array, c: number, h: number, wd: number, out: Float32Array): [number, number] {
    const oh = h >> 1;
    const ow = wd >> 1;
    for (let ci = 0; ci < c; ci++) {
      const inBase = ci * h * wd;
      const outBase = ci * oh * ow;
      for (let y = 0; y < oh; y++) {
        const r0 = inBase + (y * 2) * wd;
        const r1 = r0 + wd;
        const oRow = outBase + y * ow;
        for (let x = 0; x < ow; x++) {
          const a = input[r0 + x * 2];
          const b = input[r0 + x * 2 + 1];
          const cc = input[r1 + x * 2];
          const d = input[r1 + x * 2 + 1];
          let m = a > b ? a : b;
          if (cc > m) m = cc;
          if (d > m) m = d;
          out[oRow + x] = m;
        }
      }
    }
    return [oh, ow];
  }

  function predict(img: Uint8Array, size: number): Float32Array {
    const logits = new Float32Array(nClasses);
    if (size !== 32) throw new Error(`输入尺寸必须是 32，收到 ${size}`);

    // 输入归一化到 [0,1]
    const total = size * size;
    let cur = bufA;
    for (let i = 0; i < total; i++) cur[i] = img[i] / 255;

    let c = 1;
    let h = size;
    let wd = size;

    for (let bi = 0; bi < layers.length; bi++) {
      const layer = layers[bi];
      const dst = cur === bufA ? bufB : bufA;
      conv3x3(cur, layer, h, wd, dst);
      cur = dst;
      c = layer.cout;
      if (bi < layers.length - 1) {
        const pooled = cur === bufA ? bufB : bufA;
        const [oh, ow] = maxpool2(cur, c, h, wd, pooled);
        cur = pooled;
        h = oh;
        wd = ow;
      }
    }

    // GAP
    const area = h * wd;
    const gap = new Float32Array(c);
    for (let ci = 0; ci < c; ci++) {
      const base = ci * area;
      let s = 0;
      for (let i = 0; i < area; i++) s += cur[base + i];
      gap[ci] = s / area;
    }

    // FC
    for (let k = 0; k < nClasses; k++) {
      let s = headB[k];
      const base = k * c;
      for (let i = 0; i < c; i++) s += headW[base + i] * gap[i];
      logits[k] = s;
    }
    return logits;
  }

  function softmaxTopK(logits: Float32Array, k: number): { index: number; p: number }[] {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
    let sum = 0;
    const exps = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      const v = Math.exp(logits[i] - max);
      exps[i] = v;
      sum += v;
    }
    const idx = Array.from({ length: logits.length }, (_, i) => i);
    idx.sort((a, b) => exps[b] - exps[a]);
    return idx.slice(0, k).map((i) => ({ index: i, p: exps[i] / sum }));
  }

  return {
    classes,
    predict,
    topK: (img, size, k) => softmaxTopK(predict(img, size), k),
  };
}

/** 从打包进来的 JSON（含 base64 权重）加载模型 */
export function loadModel(payload: {
  weights: string;
  classes: ModelClass[];
}): DoodleModel {
  return parseModel(decodeBase64(payload.weights), payload.classes);
}
