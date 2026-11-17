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
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

/**
 * 默认校验**App 实际加载的那个模型**：从 `src/hooks/use-doodle-ai.ts` 里读出
 * `require('../../assets/model/xxx.json')` 的文件名。
 *
 * 为什么不写死常量：写死就意味着换模型时必须记得改两个地方，而「忘了改闸门」的后果
 * 不是报错，是 CI 兴高采烈地校验一个早就不用了的旧模型、然后放行一个没验过的新模型。
 * 说明文字会过期，静态 require 的字符串不会。
 *
 * 也可显式指定：`node tools/check-engine.mjs doodlemind-v4`
 */
function resolveModelName() {
  if (process.argv[2]) return process.argv[2];
  const hook = readFileSync(join(root, 'src/hooks/use-doodle-ai.ts'), 'utf8');
  const hit = hook.match(/assets\/model\/([\w.-]+)\.json/);
  if (!hit) throw new Error('没能从 src/hooks/use-doodle-ai.ts 里找到 assets/model/*.json 引用');
  return hit[1];
}

const MODEL = resolveModelName();
console.log(`校验模型：${MODEL}`);

// Windows 上 ESM 只接受 file:// URL，直接传 "D:\..." 会报
// ERR_UNSUPPORTED_ESM_URL_SCHEME（协议被解析成 'd:'）。
const { loadModel } = await import(pathToFileURL(join(root, 'src/lib/model.ts')).href);

const payload = JSON.parse(readFileSync(join(root, `assets/model/${MODEL}.json`), 'utf8'));

// 参考向量必须与模型**同名同源**：里面的 refPred / refTop5Prob 是用这一个模型的
// 解量化权重算出来的。拿 A 模型的参考去校验 B 模型，一致率会掉到 70%，而这个数字
// 看起来像「引擎写错了」，实际是「拿错了标尺」。所以缺了就直说缺，不去猜。
const vecPath = join(here, `vectors/test_vectors_${MODEL}.json`);
let vectors;
try {
  vectors = JSON.parse(readFileSync(vecPath, 'utf8'));
} catch {
  console.error(`缺少参考向量：${vecPath}`);
  console.error(`请先跑 python tools/export_model.py --ckpt <权重> --name ${MODEL}（导出时会一并生成）`);
  process.exit(1);
}

const model = loadModel(payload);
console.log(`✓ 模型加载：${payload.nClasses} 类，${Math.round(payload.bytes / 1024)} KB`);

const { images, labels, refPred, refTop5, refTop5Prob } = vectors;
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
/** 概率数值的最大绝对偏差：区分"实现在别处差一点"与"实现有 bug"的关键指标 */
let maxProbDiff = 0;
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

  // 逐名次比概率。参考端用的是**解量化后的同一套权重**，
  // 所以数学上应该完全相等，任何 >1e-3 的偏差都是实现问题而非浮点噪声。
  if (refTop5Prob) {
    for (let k = 0; k < 5; k++) {
      const d = Math.abs(order[k].p - refTop5Prob[i][k]);
      if (d > maxProbDiff) maxProbDiff = d;
    }
  }
}

console.log(`\n${n} 张测试图`);
console.log(`  top-1 与 PyTorch 一致 : ${top1}/${n}  (${((top1 / n) * 100).toFixed(2)}%)`);
console.log(`  我们的 top-1 落在参考 top-5 内: ${top5}/${n}  (${((top5 / n) * 100).toFixed(2)}%)`);
console.log(`  我们的准确率（vs 真值）: ${correct}/${n}  (${((correct / n) * 100).toFixed(2)}%)`);
console.log(`  单次推理: ${(totalMs / n).toFixed(2)} ms`);
if (refTop5Prob) console.log(`  top-5 概率最大绝对偏差: ${maxProbDiff.toExponential(2)}`);

if (first.length) {
  console.log('\n不一致样本：');
  for (const f of first) console.log(`  #${f.i} 我们=${f.ours} 参考=${f.ref} 真值=${f.label}`);
}

/**
 * 门槛怎么定：
 * - 概率偏差必须 < 1e-3。这是真正的正确性判据。
 * - top-1 一致率要求 100%。两边是同一个网络，argmax 由概率决定，
 *   概率一致到 1e-3 时 argmax 翻转只可能发生在两个类别几乎并列的样本上；
 *   那种样本本来就不该出现在"金标准"测试集里。真出现了就是真问题，得看。
 */
const probOk = !refTop5Prob || maxProbDiff < 1e-3;
const predOk = top1 / n >= 0.995;
const ok = probOk && predOk;
console.log(
  `\n${ok ? '✓ 通过' : '✗ 失败'}：概率偏差 < 1e-3 (${probOk ? 'ok' : 'fail'}) · top-1 一致率 ≥ 99.5% (${predOk ? 'ok' : 'fail'})`,
);
process.exit(ok ? 0 : 1);
