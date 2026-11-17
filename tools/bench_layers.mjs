// 逐层拆开量单次 forward 的耗时，搞清楚预算到底花在哪。
// 只看总时长没用：优化一个占 8% 的层，改十遍也感觉不到。
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const MODEL = process.argv[2] || 'doodlemind-v3';
const { loadModel } = await import(pathToFileURL(join(root, 'src/lib/model.ts')).href);

const payload = JSON.parse(readFileSync(join(root, `assets/model/${MODEL}.json`), 'utf8'));
const model = loadModel(payload);

// 拿一张有墨的图，别用全黑（全零输入某些路径会被优化掉，量出来的时间偏乐观）
const img = new Uint8Array(32 * 32);
for (let y = 8; y < 24; y++) for (let x = 8; x < 24; x++) img[y * 32 + x] = 200;

const N = 60;
const times = [];
for (let i = 0; i < N; i++) {
  const t = performance.now();
  model.predict(img, 32);
  times.push(performance.now() - t);
}
times.sort((a, b) => a - b);
const med = times[times.length >> 1];
const p90 = times[Math.floor(times.length * 0.9)];

console.log(`模型 ${MODEL}  ${payload.nClasses} 类  ${Math.round(payload.bytes / 1024)} KB`);
console.log(`单次推理  中位 ${med.toFixed(2)} ms   p90 ${p90.toFixed(2)} ms   最小 ${times[0].toFixed(2)} ms`);

// 理论 MAC 数：拿来和实测对，能看出是算力瓶颈还是内存瓶颈
const W = payload.width;
let macs = 0;
let h = 32;
let cin = 1;
const rows = [];
for (let i = 0; i < W.length; i++) {
  const cout = W[i];
  const m = cout * cin * 9 * h * h;
  macs += m;
  rows.push({ layer: `block${i + 1}`, shape: `${h}x${h} ${cin}->${cout}`, macs: m });
  cin = cout;
  h >>= 1;
}
rows.push({ layer: 'head', shape: `${cin}->${payload.nClasses}`, macs: cin * payload.nClasses });
macs += cin * payload.nClasses;

console.log(`\n总 MAC ${(macs / 1e6).toFixed(2)} M   实测 ${med.toFixed(2)} ms   → ${(macs / 1e6 / med).toFixed(2)} GMAC/s`);
console.log(`\n${'层'.padEnd(10)} ${'形状'.padEnd(18)} ${'MAC'.padStart(9)}  占比`);
for (const r of rows) {
  console.log(
    `${r.layer.padEnd(10)} ${r.shape.padEnd(18)} ${(r.macs / 1e6).toFixed(2).padStart(7)}M  ${((r.macs / macs) * 100).toFixed(1).padStart(5)}%`,
  );
}
