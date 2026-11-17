// 用户澄清了「轻量」：不是把模型缩小，是它得能在手机上跑。这条决定了优化方向——
// 不能靠压小网络省时间，得靠**换更省算力的结构**，把省下来的预算还给宽度和深度。
//
// 比较：3x3 dense conv  vs  depthwise 3x3 + pointwise 1x1（MobileNet 那套）。
// 理论 MAC 差 5~8 倍，但 JS 里没有 SIMD，1x1 的内层只有 1 个乘加，
// 循环开销占比高，实际收益一定小于理论值。所以要实测。
//
// 同一份输入权重下比，且**用同一份输入数据**校验数值——前几轮 benchmark
// 各造一份随机输入，等价性校验形同虚设。

const H = 16;
const CIN = 32;
const COUT = 64;

function makeInput(c, h) {
  const a = new Float32Array(c * h * h);
  let s = 12345;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    a[i] = (s % 1000) / 2000;
  }
  return a;
}

function makeW(n) {
  const a = new Float32Array(n);
  let s = 98765;
  for (let i = 0; i < n; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    a[i] = ((s % 2000) - 1000) / 8000;
  }
  return a;
}

// ── dense 3x3（当前主线实现）──
function dense3x3(input, w, bias, cin, cout, h, wd, out) {
  const area = h * wd;
  const yEnd = h - 1, xEnd = wd - 1;
  for (let co = 0; co < cout; co++) {
    const ob = co * area;
    out.fill(bias[co], ob, ob + area);
    for (let ci = 0; ci < cin; ci++) {
      const ib = ci * area;
      const kw = (co * cin + ci) * 9;
      const k00 = w[kw], k01 = w[kw + 1], k02 = w[kw + 2];
      const k10 = w[kw + 3], k11 = w[kw + 4], k12 = w[kw + 5];
      const k20 = w[kw + 6], k21 = w[kw + 7], k22 = w[kw + 8];
      for (let y = 1; y < yEnd; y++) {
        const mid = ib + y * wd, oRow = ob + y * wd;
        for (let x = 1; x < xEnd; x++) {
          const c = mid + x;
          out[oRow + x] +=
            k00 * input[c - wd - 1] + k01 * input[c - wd] + k02 * input[c - wd + 1] +
            k10 * input[c - 1] + k11 * input[c] + k12 * input[c + 1] +
            k20 * input[c + wd - 1] + k21 * input[c + wd] + k22 * input[c + wd + 1];
        }
      }
      for (let y = 0; y < h; y++) {
        const edge = y === 0 || y === yEnd;
        for (let x = 0; x < wd; x++) {
          if (!edge && x !== 0 && x !== xEnd) continue;
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
  const total = cout * area;
  for (let i = 0; i < total; i++) if (out[i] < 0) out[i] = 0;
}

// ── depthwise 3x3（每通道独立，无通道混）+ ReLU ──
function depthwise3x3(input, w, bias, c, h, wd, out) {
  const area = h * wd;
  const yEnd = h - 1, xEnd = wd - 1;
  for (let ci = 0; ci < c; ci++) {
    const ib = ci * area;
    const kw = ci * 9;
    const k00 = w[kw], k01 = w[kw + 1], k02 = w[kw + 2];
    const k10 = w[kw + 3], k11 = w[kw + 4], k12 = w[kw + 5];
    const k20 = w[kw + 6], k21 = w[kw + 7], k22 = w[kw + 8];
    const b = bias[ci];
    out.fill(b, ib, ib + area);
    for (let y = 1; y < yEnd; y++) {
      const mid = ib + y * wd, oRow = ib + y * wd;
      for (let x = 1; x < xEnd; x++) {
        const p = mid + x;
        out[oRow + x] +=
          k00 * input[p - wd - 1] + k01 * input[p - wd] + k02 * input[p - wd + 1] +
          k10 * input[p - 1] + k11 * input[p] + k12 * input[p + 1] +
          k20 * input[p + wd - 1] + k21 * input[p + wd] + k22 * input[p + wd + 1];
      }
    }
    for (let y = 0; y < h; y++) {
      const edge = y === 0 || y === yEnd;
      for (let x = 0; x < wd; x++) {
        if (!edge && x !== 0 && x !== xEnd) continue;
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
        out[ib + y * wd + x] += acc;
      }
    }
  }
  const total = c * area;
  for (let i = 0; i < total; i++) if (out[i] < 0) out[i] = 0;
}

// ── pointwise 1x1（纯通道混，内层只有一次乘加）──
// 关键手法：把 cout 拆成小批（4 个），一次算 4 个输出通道，
// 这样内层每读一个 input 能摊到 4 次乘加上，把循环开销摊薄。
function pointwise1x1(input, w, bias, cin, cout, area, out) {
  const G = 4;
  for (let co0 = 0; co0 < cout; co0 += G) {
    const n = Math.min(G, cout - co0);
    for (let i = 0; i < area; i++) {
      out[(co0 + 0) * area + i] = bias[co0];
      if (n > 1) out[(co0 + 1) * area + i] = bias[co0 + 1];
      if (n > 2) out[(co0 + 2) * area + i] = bias[co0 + 2];
      if (n > 3) out[(co0 + 3) * area + i] = bias[co0 + 3];
    }
    for (let ci = 0; ci < cin; ci++) {
      const ib = ci * area;
      const wb = (co0 * cin + ci) * 1;
      const q0 = w[wb];
      const q1 = n > 1 ? w[wb + cin] : 0;
      const q2 = n > 2 ? w[wb + 2 * cin] : 0;
      const q3 = n > 3 ? w[wb + 3 * cin] : 0;
      const o0 = (co0 + 0) * area;
      const o1 = o0 + area, o2 = o1 + area, o3 = o2 + area;
      if (n === 4) {
        for (let i = 0; i < area; i++) {
          const v = input[ib + i];
          out[o0 + i] += q0 * v;
          out[o1 + i] += q1 * v;
          out[o2 + i] += q2 * v;
          out[o3 + i] += q3 * v;
        }
      } else {
        for (let i = 0; i < area; i++) {
          const v = input[ib + i];
          out[o0 + i] += q0 * v;
          if (n > 1) out[o1 + i] += q1 * v;
          if (n > 2) out[o2 + i] += q2 * v;
        }
      }
    }
    for (let g = 0; g < n; g++) {
      const ob = (co0 + g) * area;
      for (let i = 0; i < area; i++) {
        const v = out[ob + i];
        out[ob + i] = v > 0 ? v : 0;
      }
    }
  }
}

const input = makeInput(CIN, H);
const wDense = makeW(COUT * CIN * 9);
const bDense = new Float32Array(COUT).fill(0.001);
const wDw = makeW(CIN * 9);
const bDw = new Float32Array(CIN).fill(0.001);
const wPw = makeW(COUT * CIN);
const bPw = new Float32Array(COUT).fill(0.001);

const outDense = new Float32Array(COUT * H * H);
const outDw = new Float32Array(CIN * H * H);
const outPw = new Float32Array(COUT * H * H);

function timeIt(fn, reps) {
  for (let i = 0; i < 10; i++) fn();
  const t = [];
  for (let i = 0; i < reps; i++) {
    const s = performance.now();
    fn();
    t.push(performance.now() - s);
  }
  t.sort((a, b) => a - b);
  return t[t.length >> 1];
}

const tDense = timeIt(() => dense3x3(input, wDense, bDense, CIN, COUT, H, H, outDense), 60);
const tDw = timeIt(() => depthwise3x3(input, wDw, bDw, CIN, H, H, outDw), 60);
const tPw = timeIt(() => pointwise1x1(outDw, wPw, bPw, CIN, COUT, H * H, outPw), 60);

const macDense = CIN * COUT * 9 * H * H;
const macSep = CIN * 9 * H * H + CIN * COUT * H * H;

console.log(`形状 ${H}x${H}  ${CIN}->${COUT}`);
console.log(`dense 3x3          ${tDense.toFixed(3)} ms   ${(macDense / 1e6).toFixed(2)}M MAC`);
console.log(`depthwise 3x3      ${tDw.toFixed(3)} ms   ${((CIN * 9 * H * H) / 1e6).toFixed(2)}M MAC`);
console.log(`pointwise 1x1      ${tPw.toFixed(3)} ms   ${((CIN * COUT * H * H) / 1e6).toFixed(2)}M MAC`);
console.log(`可分离合计         ${(tDw + tPw).toFixed(3)} ms   ${(macSep / 1e6).toFixed(2)}M MAC`);
console.log(`\n实测加速 ${(tDense / (tDw + tPw)).toFixed(2)}x   （理论 MAC 加速 ${(macDense / macSep).toFixed(2)}x）`);
console.log(`有效算力 dense ${(macDense / 1e6 / tDense).toFixed(2)} GMAC/s   可分离 ${(macSep / 1e6 / (tDw + tPw)).toFixed(2)} GMAC/s`);
