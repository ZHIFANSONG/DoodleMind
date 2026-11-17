/**
 * 栅格化一致性闸门。
 *
 * `tools/check-engine.mjs` 只验证了**推理**——它吃的是已经栅格化好的 uint8 图。
 * 但端上真正跑的路径是「原始笔画 → 栅格化 → 推理」，前半段一直没人看过。
 *
 * 这一段的 bug 极其阴险：32x32 输出是灰度的，画偏了也只是内容跑到角落、
 * 被裁掉一半，不会报错，模型只是悄悄变笨。所以必须拿 Python 的结果逐字节比。
 *
 * 跑：node tools/check-raster.mjs
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const { rasterize, RASTER_SIZE } = await import(pathToFileURL(join(root, 'src/lib/raster.ts')).href);

const data = JSON.parse(readFileSync(join(root, 'tools/out/raster_vectors.json'), 'utf8'));
if (data.size !== RASTER_SIZE) {
  console.error(`✗ size 不一致：python=${data.size} ts=${RASTER_SIZE}`);
  process.exit(1);
}

let exact = 0;
let near = 0;
let worst = { diff: 0, word: '' };
const bad = [];

for (const v of data.vectors) {
  const got = rasterize(v.strokes, RASTER_SIZE);
  let maxDiff = 0;
  let sumAbs = 0;
  for (let i = 0; i < got.length; i++) {
    const d = Math.abs(got[i] - v.expected[i]);
    sumAbs += d;
    if (d > maxDiff) maxDiff = d;
  }
  if (maxDiff === 0) exact++;
  // 允许 1 个灰度级的差：Python 用 round()，JS 的 Math.round 在 .5 处的
  // 舍入方向也是「向正无穷」，两边理论上一致；容 1 级是为了防浮点尾数。
  if (maxDiff <= 1) near++;
  else bad.push({ word: v.word, maxDiff, sumAbs });

  if (maxDiff > worst.diff) worst = { diff: maxDiff, word: v.word };
}

const n = data.vectors.length;
console.log(`\n${n} 组笔画（含 3 组人工构造的偏心/极小样本）`);
console.log(`  逐像素完全一致 : ${exact}/${n}`);
console.log(`  最大偏差 ≤ 1   : ${near}/${n}`);
console.log(`  最大灰度偏差   : ${worst.diff} (${worst.word})`);

if (bad.length) {
  console.log('\n超差样本：');
  for (const b of bad) console.log(`  ${b.word}: maxDiff=${b.maxDiff} sumAbs=${b.sumAbs}`);
}

/**
 * 平移不变性：同一个图形，画在画布左上角和右下角，栅格化结果必须完全相同。
 *
 * 这是**独立于参考数据的**判据，比逐像素比对更能锁住那个具体的 bug。
 * 原因是 QuickDraw 数据几乎都贴着左上角（实测 88.6% 的样本 min_x 正好是 0），
 * 光靠它测不出「缩放前忘了减包围盒原点」——那个 bug 在训练侧根本不会暴露，
 * 只会在玩家把手移到屏幕中间时让模型突然变哑。这条断言把它钉死。
 */
let worstShift = 0;
for (const v of data.vectors) {
  const shifted = v.strokes.map(([xs, ys]) => [xs.map((x) => x + 137), ys.map((y) => y + 91)]);
  const a = rasterize(v.strokes, RASTER_SIZE);
  const b = rasterize(shifted, RASTER_SIZE);
  for (let i = 0; i < a.length; i++) worstShift = Math.max(worstShift, Math.abs(a[i] - b[i]));
}
console.log(`  平移不变性     : 最大灰度差 ${worstShift}`);

const ok = near === n && worstShift === 0;
console.log(
  `\n${ok ? '✓ 通过' : '✗ 失败'}：要求全部样本最大灰度偏差 ≤ 1，且平移不改变结果`,
);
process.exit(ok ? 0 : 1);
