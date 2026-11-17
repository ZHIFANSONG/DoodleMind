/**
 * 端上引擎一致性校验。
 *
 * 目的：证明「TypeScript 手写前向传播」和「PyTorch 参考实现」在同样输入上给出同样答案。
 * 这是整个项目最容易悄悄坏掉的一环——BN 折叠写错、量化 scale 错位、卷积下标转置，
 * 任何一个都会让模型静默变成「一个总能猜出猫的模型」，而 App 照样跑得好好的。
 *
 * 用法：node tools/check-engine.mjs
 * 依赖：Node 22.18+ / 24（原生 TS 类型剥离），无需 tsx / esbuild。
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const { loadModel } = await import(join(root, 'src/lib/model.ts'));

const payload = JSON.parse(readFileSync(join(root, 'assets/model/doodlemind-v1.json'), 'utf8'));
const vectors = JSON.parse(readFileSync(join(here, 'out/test_vectors.json'), 'utf8'));

const model = loadModel(payload);
console.log(`✓ 模型加载：${payload.nClasses} 类，${Math.round(payload.bytes / 1024)} KB`);

const { images, labels, refPred, refTop5 } = vectors;
const n = images.length;

function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const v = Math.exp(logits[i] - max);
    out[i] = v;
    sum += v;
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

let top1 = 0;
let top5 = 0;
let correct = 0;
let totalMs = 0;
const first = [];

for (let i = 0; i < n; i++) {
  const img = Uint8Array.from(images[i]);

  const t0 = performance.now();
  const logits = model.predict(img, 32);
  totalMs += performance.now() - t0;

  const probs = softmax(logits);
  const order = Array.from(probs, (p, index) => ({ index, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 5);

  const pred = order[0].index;
  if (pred === refPred[i]) top1++;
  else if (first.length < 5) first.push({ i, ours: pred, ref: refPred[i], label: labels[i] });

  if (refTop5[i].includes(pred)) top5++;
  if (pred === labels[i]) correct++;
}

console.log(`\n${n} 张测试图`);
console.log(`  top-1 与 PyTorch 一致 : ${top1}/${n}  (${((top1 / n) * 100).toFixed(2)}%)`);
console.log(`  我们的 top-1 落在参考 top-5 内: ${top5}/${n}  (${((top5 / n) * 100).toFixed(2)}%)`);
console.log(`  我们的准确率（vs 真值）: ${correct}/${n}  (${((correct / n) * 100).toFixed(2)}%)`);
console.log(`  单次推理: ${(totalMs / n).toFixed(2)} ms`);

if (first.length) {
  console.log('\n不一致样本：');
  for (const f of first) console.log(`  #${f.i} 我们=${f.ours} 参考=${f.ref} 真值=${f.label}`);
}

// 门槛：一致率低于 99.5% 说明实现有 bug，而不是「浮点误差」
const ok = top1 / n >= 0.995;
console.log(`\n${ok ? '✓ 通过' : '✗ 失败'}：要求 top-1 一致率 ≥ 99.5%`);
process.exit(ok ? 0 : 1);
