/**
 * 端上 CNN 推理引擎。纯 TypeScript，无任何原生依赖、无 tfjs。
 *
 * 网络结构由模型文件里的 op 序列决定（v4 格式），支持三种算子混搭：
 *   dense3x3 / depthwise3x3 / pointwise1x1 —— 后两者组合起来就是可分离卷积。
 * 训练侧 `tools/train.py --arch sep` 产出的结构：
 *   输入 1x32x32
 *   dense3x3(1→48)  +ReLU  @32 →pool→16
 *   depthwise(48)   +ReLU  @16
 *   pointwise(48→160)+ReLU @16 →pool→8
 *   depthwise(160)  +ReLU  @8
 *   pointwise(160→320)+ReLU@8  →pool→4
 *   depthwise(320)  +ReLU  @4
 *   pointwise(320→512)+ReLU@4
 *   GAP → FC(512→N) → softmax
 *
 * 为什么值得多写两种算子：端上引擎是纯 JS，没有 SIMD，实测有效算力约
 * 1.0~1.2 GMAC/s，而这张网每帧都要跑一次（玩家边画边判）。也就是说
 * **模型的强度上限被 MAC 数锁死**。可分离卷积把每个 MAC 的价值提到 7 倍
 * （实测 6.92x，几乎吃满理论值），省下的算力全部还给通道数——
 * 这是在不牺牲手感的前提下把模型做强的唯一途径。
 *
 * 权重以 per-channel 对称 int8 存储，首次加载时一次性解量化成
 * Float32Array 常驻内存。推理全程 float32 —— JS 里没有 SIMD，
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

type OpType = 0 | 1 | 2; // 0=dense3x3  1=depthwise3x3  2=pointwise1x1

type Layer = {
  type: OpType;
  cin: number;
  cout: number;
  /** 该层之后是否接 2x2 maxpool */
  pool: boolean;
  /**
   * 解量化后的 float32 权重，行主序。元素个数由 type 决定：
   *   dense3x3    cout*cin*9   [co][ci][ky][kx]
   *   depthwise   cin*9        [ci][ky][kx]（每通道独享一组 3x3）
   *   pointwise   cout*cin     [co][ci]
   */
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
  // v2：分类头是 f32；v3：分类头也做了 per-class int8 量化；
  // v4：每个 op 前带类型字节（dense/dw/pw），支持可分离卷积结构。
  // 全部认，免得旧权重文件因为格式升级当场报废。
  if (version < 2 || version > 4) throw new Error(`模型版本 ${version} 不支持（期望 2~4）`);
  const nClasses = view.getUint16(p, true);
  p += 2;
  const nBlocks = view.getUint16(p, true);
  p += 2;
  if (nClasses !== classes.length) {
    throw new Error(`类别数不一致：模型 ${nClasses} vs 词表 ${classes.length}`);
  }

  const layers: Layer[] = [];
  for (let bi = 0; bi < nBlocks; bi++) {
    // v4 才带 type/flags；旧格式一律视作 dense3x3、前 3 层池化。
    let type: OpType = 0;
    let pool = bi < nBlocks - 1;
    if (version >= 4) {
      type = view.getUint8(p++) as OpType;
      const flags = view.getUint8(p++);
      pool = (flags & 1) === 1;
    }
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

    // 权重个数按算子类型算。这三条公式必须和 tools/export_model.py::weight_count 一致，
    // 差一个数就会读到下一层的字节，不报错，只是模型悄悄变笨。
    const n = type === 0 ? cout * cin * 9 : type === 1 ? cin * 9 : cout * cin;
    const w = new Float32Array(n);
    let q = p;
    let oi = 0;
    if (type === 0 || type === 2) {
      // 两种都是 per-cout 的 scale：dense 每组 cin*9 个，pointwise 每组 cin 个
      const per = type === 0 ? cin * 9 : cin;
      for (let co = 0; co < cout; co++) {
        const s = wScale[co];
        for (let t = 0; t < per; t++) w[oi++] = view.getInt8(q++) * s;
      }
    } else {
      // depthwise 每组只有 9 个权重，但「组」是输入通道，scale 也是按输入通道给的
      for (let ci = 0; ci < cin; ci++) {
        const s = wScale[ci];
        for (let t = 0; t < 9; t++) w[oi++] = view.getInt8(q++) * s;
      }
    }
    p += n;
    layers.push({ type, cin, cout, pool, w, bias });
  }

  const width = view.getUint16(p, true);
  p += 2;
  const headW = new Float32Array(nClasses * width);
  if (version >= 3) {
    // v3：每类一组 scale，权重是 int8。解量化回 f32 再算，
    // 因为激活本来就是 f32，点积保持同一量纲最省心（head 只有 19k 次乘加）。
    const headScale = new Float32Array(nClasses);
    for (let i = 0; i < nClasses; i++) {
      headScale[i] = view.getFloat32(p, true);
      p += 4;
    }
    let q = p;
    let oi = 0;
    for (let k = 0; k < nClasses; k++) {
      const s = headScale[k];
      for (let i = 0; i < width; i++) {
        headW[oi++] = view.getInt8(q) * s;
        q += 1;
      }
    }
    p += nClasses * width;
  } else {
    for (let i = 0; i < headW.length; i++) {
      headW[i] = view.getFloat32(p, true);
      p += 4;
    }
  }
  const headB = new Float32Array(nClasses);
  for (let i = 0; i < nClasses; i++) {
    headB[i] = view.getFloat32(p, true);
    p += 4;
  }

  // 复用的中间缓冲。尺寸必须盖住**每一层**的通道数 × 面积，
  // 不能只看 cout：depthwise 的输出通道数等于 cin，而它出现在面积更大的层上。
  // 分配一次常驻，避免每帧 GC。
  let maxCells = 0;
  {
    let hh = 32;
    for (const l of layers) {
      const cells = Math.max(l.cin, l.cout) * hh * hh;
      if (cells > maxCells) maxCells = cells;
      if (l.pool) hh >>= 1;
    }
  }
  let bufA = new Float32Array(maxCells);
  let bufB = new Float32Array(maxCells);

  /**
   * 3x3 pad=1 conv + ReLU。in: (cin,h,w) 行主序。
   *
   * 循环顺序是**输出通道在外、输入通道在内**，这跟直觉相反但很要紧：
   * 反过来的话（输入通道在外、输出通道在内）每算一个 tap 就要把所有 cout 个
   * 输出平面各摸一次，32x32x128 的输出是 512KB，远超 L2，等于每次都在内存里
   * 来回跳。改成通道在外之后，累加只落在一个 4KB 的平面上，全程留在缓存里。
   *
   * 另外把内部区域和边缘拆开：边缘要判越界，内部不用。32x32 里九成以上是内部。
   * 推理是**每一帧都要跑一次**的路径（玩家边画边判），这里省下的每一毫秒都算数。
   */
  function conv3x3(
    input: Float32Array,
    layer: Layer,
    h: number,
    wd: number,
    out: Float32Array,
  ): void {
    const { cin, cout, w, bias } = layer;
    const area = h * wd;
    const yEnd = h - 1;
    const xEnd = wd - 1;

    for (let co = 0; co < cout; co++) {
      const ob = co * area;
      const b = bias[co];
      out.fill(b, ob, ob + area);

      for (let ci = 0; ci < cin; ci++) {
        const ib = ci * area;
        const kw = (co * cin + ci) * 9;
        const k00 = w[kw];
        const k01 = w[kw + 1];
        const k02 = w[kw + 2];
        const k10 = w[kw + 3];
        const k11 = w[kw + 4];
        const k12 = w[kw + 5];
        const k20 = w[kw + 6];
        const k21 = w[kw + 7];
        const k22 = w[kw + 8];
        // int8 量化后整组为 0 的通道不罕见（死通道），跳过省一整趟
        if (k00 === 0 && k01 === 0 && k02 === 0 && k10 === 0 && k11 === 0 && k12 === 0 && k20 === 0 && k21 === 0 && k22 === 0) {
          continue;
        }

        // ── 内部区域：不需要判越界，九成以上的乘加都在这儿 ──
        for (let y = 1; y < yEnd; y++) {
          const mid = ib + y * wd;
          const oRow = ob + y * wd;
          for (let x = 1; x < xEnd; x++) {
            const c = mid + x;
            out[oRow + x] +=
              k00 * input[c - wd - 1] + k01 * input[c - wd] + k02 * input[c - wd + 1] +
              k10 * input[c - 1] + k11 * input[c] + k12 * input[c + 1] +
              k20 * input[c + wd - 1] + k21 * input[c + wd] + k22 * input[c + wd + 1];
          }
        }
      }
    }
    // ── 边界那一圈（第一/最后一行 + 第一/最后一列，共 4h-4 个像素）──
    // 一律走最朴素的判越界通式，**不给任何一条边写特例**。
    // 我先试过为左右端点手写展开（省几十次循环判断），结果左端点漏掉了
    // kx=-1 越界的那三个 tap，top-1 一致率立刻从 100% 掉到 97.5%——
    // 而这种错不会崩、不会报错，只会让模型"有点笨"。为几十个像素担这个风险不值得。
    for (let co = 0; co < cout; co++) {
      const ob = co * area;
      for (let ci = 0; ci < cin; ci++) {
        const ib = ci * area;
        const kw = (co * cin + ci) * 9;
        for (let y = 0; y < h; y++) {
          const edgeRow = y === 0 || y === yEnd;
          for (let x = 0; x < wd; x++) {
            if (!edgeRow && x !== 0 && x !== xEnd) continue;
            let acc = 0;
            for (let ky = 0; ky < 3; ky++) {
              const sy = y + ky - 1;
              if (sy < 0 || sy >= h) continue;
              for (let kx = 0; kx < 3; kx++) {
                const sx = x + kx - 1;
                if (sx < 0 || sx >= wd) continue;
                acc += w[kw + ky * 3 + kx] * input[ib + sy * wd + sx];
              }
            }
            out[ob + y * wd + x] += acc;
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

  /**
   * depthwise 3x3 pad=1 + ReLU。每通道独享一组 3x3，不做通道混合。
   *
   * 和 dense 版相比它省掉了 cin 这层循环：dense 的 MAC 是 cout*cin*9*area，
   * 这里只有 cin*9*area。代价是通道之间不再交互，必须靠紧跟的 pointwise 补回来。
   * 内部/边界照样拆开——同一套理由，别为几十个像素写特例。
   */
  function conv3x3dw(
    input: Float32Array,
    layer: Layer,
    h: number,
    wd: number,
    out: Float32Array,
  ): void {
    const { cin, w, bias } = layer;
    const area = h * wd;
    const yEnd = h - 1;
    const xEnd = wd - 1;

    for (let ci = 0; ci < cin; ci++) {
      const base = ci * area;
      const b = bias[ci];
      out.fill(b, base, base + area);
      const kw = ci * 9;
      const k00 = w[kw];
      const k01 = w[kw + 1];
      const k02 = w[kw + 2];
      const k10 = w[kw + 3];
      const k11 = w[kw + 4];
      const k12 = w[kw + 5];
      const k20 = w[kw + 6];
      const k21 = w[kw + 7];
      const k22 = w[kw + 8];
      // 整组为零的通道直接跳过：量化后这种通道不罕见
      if (k00 === 0 && k01 === 0 && k02 === 0 && k10 === 0 && k11 === 0 && k12 === 0 && k20 === 0 && k21 === 0 && k22 === 0) {
        continue;
      }

      for (let y = 1; y < yEnd; y++) {
        const mid = base + y * wd;
        const oRow = mid;
        for (let x = 1; x < xEnd; x++) {
          const c = mid + x;
          out[oRow + x] +=
            k00 * input[c - wd - 1] + k01 * input[c - wd] + k02 * input[c - wd + 1] +
            k10 * input[c - 1] + k11 * input[c] + k12 * input[c + 1] +
            k20 * input[c + wd - 1] + k21 * input[c + wd] + k22 * input[c + wd + 1];
        }
      }

      // 边界一圈：判越界通式
      for (let y = 0; y < h; y++) {
        const edgeRow = y === 0 || y === yEnd;
        for (let x = 0; x < wd; x++) {
          if (!edgeRow && x !== 0 && x !== xEnd) continue;
          let acc = 0;
          for (let ky = 0; ky < 3; ky++) {
            const sy = y + ky - 1;
            if (sy < 0 || sy >= h) continue;
            for (let kx = 0; kx < 3; kx++) {
              const sx = x + kx - 1;
              if (sx < 0 || sx >= wd) continue;
              acc += w[kw + ky * 3 + kx] * input[base + sy * wd + sx];
            }
          }
          out[base + y * wd + x] += acc;
        }
      }
    }

    const total = cin * area;
    for (let i = 0; i < total; i++) {
      if (out[i] < 0) out[i] = 0;
    }
  }

  /**
   * pointwise 1x1 + ReLU：纯通道混合，每个输出像素是 cin 个数的点积。
   *
   * 内层只有一次乘加，循环开销占比高（实测比 dense 的「有效算力」略低），
   * 所以**一次算 4 个输出通道**：读一个 input 值摊到 4 次乘加上，
   * 把下标运算和循环判断摊薄。实测这一手在 16x16 上值 4~5 成。
   */
  function conv1x1(
    input: Float32Array,
    layer: Layer,
    area: number,
    out: Float32Array,
  ): void {
    const { cin, cout, w, bias } = layer;
    const G = 4;

    for (let co0 = 0; co0 < cout; co0 += G) {
      const n = cout - co0 < G ? cout - co0 : G;
      const b0 = bias[co0];
      const o0 = co0 * area;
      const o1 = o0 + area;
      const o2 = o1 + area;
      const o3 = o2 + area;
      out.fill(b0, o0, o0 + area);
      if (n > 1) out.fill(bias[co0 + 1], o1, o1 + area);
      if (n > 2) out.fill(bias[co0 + 2], o2, o2 + area);
      if (n > 3) out.fill(bias[co0 + 3], o3, o3 + area);

      if (n === 4) {
        for (let ci = 0; ci < cin; ci++) {
          const ib = ci * area;
          const wb = (co0 * cin) + ci;
          const q0 = w[wb];
          const q1 = w[wb + cin];
          const q2 = w[wb + 2 * cin];
          const q3 = w[wb + 3 * cin];
          for (let i = 0; i < area; i++) {
            const v = input[ib + i];
            out[o0 + i] += q0 * v;
            out[o1 + i] += q1 * v;
            out[o2 + i] += q2 * v;
            out[o3 + i] += q3 * v;
          }
        }
      } else {
        for (let ci = 0; ci < cin; ci++) {
          const ib = ci * area;
          const wb = (co0 * cin) + ci;
          const q0 = w[wb];
          for (let i = 0; i < area; i++) out[o0 + i] += q0 * input[ib + i];
          if (n > 1) {
            const q1 = w[wb + cin];
            for (let i = 0; i < area; i++) out[o1 + i] += q1 * input[ib + i];
          }
          if (n > 2) {
            const q2 = w[wb + 2 * cin];
            for (let i = 0; i < area; i++) out[o2 + i] += q2 * input[ib + i];
          }
        }
      }

      // ReLU 逐组做，避免再扫一遍整个输出
      const lim = (co0 + n) * area;
      for (let i = o0; i < lim; i++) {
        if (out[i] < 0) out[i] = 0;
      }
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
      // 按算子类型分派。type 与 tools/export_model.py 的 T_* 常量一一对应。
      if (layer.type === 0) conv3x3(cur, layer, h, wd, dst);
      else if (layer.type === 1) conv3x3dw(cur, layer, h, wd, dst);
      else conv1x1(cur, layer, h * wd, dst);
      cur = dst;
      c = layer.cout;
      if (layer.pool) {
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
