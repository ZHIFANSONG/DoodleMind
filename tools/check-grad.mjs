/**
 * 第四道闸门：反向传播（X-Ray 热力图的地基）。
 *
 * 为什么这道闸门必须存在：`src/lib/model.ts` 里手写了反向传播——四种算子的转置
 * 卷积、maxpool 的 argmax 回路由、ReLU 掩码。任何一处下标写反都不会崩，只会让
 * 热力图指错地方。而"指错地方"在热力图上看起来永远都挺像那么回事，肉眼抓不到。
 *
 * ── 为什么不直接做有限差分 ──
 * 我先试过：随机方向 v，比 `Σ g·v` 和 `(f(x+εv)−f(x−εv))/2ε`。结果残差 4~20%，
 * 有个方向甚至符号反了。看着像梯度写错了，其实是**校验方法**的锅：
 *   1. 输入是 0..1，涂鸦里六成像素正好是 0（空白）或 1（实心墨）。±ε 再 clamp
 *      会让一侧扰动被吃掉，差分退化成单边近似。
 *   2. 更麻烦的是 ReLU 折点：一次扰动会跨过成百上千个 ReLU 的折点，f 在那儿
 *      本来就不可导，差分给出的是"平均斜率"，和解析导数天然对不上几个百分点。
 * 所以改走和前三道闸门一样的路子：**两个独立实现算同一个数**。PyTorch autograd
 * 当参考（`tools/make_grad_vectors.py`），端上手写反传去比。
 *
 * 实测：cos = 1.00000，逐元素最大差 0.0000，logits 差 4e-6 —— 手写反传是精确的。
 */
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { registerHooks } from 'node:module';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.startsWith('.') && !/\.[cm]?[jt]sx?$/.test(specifier)) {
      return nextResolve(`${specifier}.ts`, context);
    }
    return nextResolve(specifier, context);
  },
});

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const { loadModel } = await import(pathToFileURL(join(root, 'src/lib/model.ts')).href);
const { rasterize } = await import(pathToFileURL(join(root, 'src/lib/raster.ts')).href);

const payload = JSON.parse(readFileSync(join(root, 'assets/model/doodlemind-v4.json'), 'utf8'));
const model = loadModel({ weights: payload.weights, classes: payload.classes });
const fixture = JSON.parse(readFileSync(join(root, 'tools/vectors/grad_vectors.json'), 'utf8'));

const SIZE = fixture.size ?? 32;
let fails = 0;
const check = (name, ok, detail) => {
  if (!ok) fails++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? `  ${detail}` : ''}`);
};

const COS_MIN = 0.9995;
let worstCos = 1;
let worstGrad = 0;
let worstLogit = 0;
const t0 = Date.now();
let runs = 0;

for (const [i, v] of fixture.vectors.entries()) {
  const img = rasterize(v.strokes, SIZE);
  const x0 = new Float32Array(SIZE * SIZE);
  for (let k = 0; k < x0.length; k++) x0[k] = img[k] / 255;

  // 先比 logits：对不上说明这份 fixture 和端上模型不是同一个网络，
  // 那就不该继续拿它比梯度，得先修 fixture。
  const logits = model.forwardNormalized(x0, SIZE);
  let logitDiff = 0;
  for (let k = 0; k < logits.length; k++) {
    logitDiff = Math.max(logitDiff, Math.abs(logits[k] - v.logits[k]));
  }
  worstLogit = Math.max(worstLogit, logitDiff);

  const grad = model.gradInput(img, SIZE, v.cls);
  runs++;
  if (!grad) {
    check(`样例 ${i} 梯度可算`, false, '返回 null');
    continue;
  }

  let dot = 0;
  let na = 0;
  let nb = 0;
  let maxDiff = 0;
  for (let k = 0; k < grad.length; k++) {
    dot += grad[k] * v.grad[k];
    na += grad[k] * grad[k];
    nb += v.grad[k] * v.grad[k];
    maxDiff = Math.max(maxDiff, Math.abs(grad[k] - v.grad[k]));
  }
  const cos = dot / Math.sqrt(na * nb);
  worstCos = Math.min(worstCos, cos);
  worstGrad = Math.max(worstGrad, maxDiff);
  console.log(
    `  样例 ${i} ${v.word.padEnd(6)} → ${String(v.clsEn).padEnd(8)} cos ${cos.toFixed(6)}  逐元素最大差 ${maxDiff.toExponential(2)}  logits 差 ${logitDiff.toExponential(2)}`,
  );
}

check('fixture 与端上模型是同一个网络（logits 差 < 1e-4）', worstLogit < 1e-4, `worst=${worstLogit.toExponential(2)}`);
check(`梯度方向与 PyTorch autograd 一致（cos ≥ ${COS_MIN}）`, worstCos >= COS_MIN, `worst=${worstCos.toFixed(6)}`);
check('梯度逐元素数值一致（最大差 < 1e-3）', worstGrad < 1e-3, `worst=${worstGrad.toExponential(2)}`);

const ms = (Date.now() - t0) / runs;
console.log(`\n  平均每次反向传播 ${ms.toFixed(2)} ms（${runs} 次，含栅格化）`);

console.log(fails === 0 ? '\n✓ 通过：手写反向传播与 autograd 一致' : `\n✗ ${fails} 项未通过`);
process.exit(fails === 0 ? 0 : 1);
