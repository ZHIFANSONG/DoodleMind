/**
 * 玩法状态机闸门。
 *
 * 为什么单独给它一道闸：判定规则是三种玩法唯一的差别，也是这个游戏成不成立的地基。
 * 它没有 UI、没有动画、纯函数式推进，因此可以完全脱离手机在 Node 里跑——
 * 这正是最该被逐条钉死的东西。人肉在真机上试「盲区模式超时到底算赢还是算平」
 * 每次要等 40 秒，试十次就废掉一个下午，而且还容易记错。
 *
 * 用法：node tools/check-game.mjs
 */

import { registerHooks } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';

// Node 能直接跑 .ts（类型剥离），但它按 ESM 严格解析，不做扩展名补全。
// 源码里写的是 import { ... } from './game'（Metro 帮我们补 .ts），
// 到了 Node 就变成 ERR_MODULE_NOT_FOUND。这里补一个 resolve 钩子。
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.startsWith('.') && !/\.[cm]?[jt]sx?$/.test(specifier)) {
      try {
        return nextResolve(`${specifier}.ts`, context);
      } catch {
        /* 落回原样，让真正的错误暴露出来 */
      }
    }
    return nextResolve(specifier, context);
  },
});

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const { Session } = await import(pathToFileURL(join(root, 'src/lib/session.ts')).href);
const { MODES } = await import(pathToFileURL(join(root, 'src/lib/modes.ts')).href);

const N = 148;
let failures = 0;

function check(name, cond, detail) {
  if (cond) {
    console.log(`  ok   ${name}`);
  } else {
    failures++;
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

/** 造一个「目标类极高、其它类平分」的概率分布 */
function probsFor(target, p) {
  const arr = new Float32Array(N).fill((1 - p) / (N - 1));
  arr[target] = p;
  return arr;
}

// ---------------------------------------------------------------- 挑战模式
console.log('\n挑战模式');
{
  const s = new Session('challenge', 7, N);
  s.start(1000);
  // 目标概率够高且排第一，但要连续保持 380ms 才算数
  check('首帧不移交终局', s.feed(probsFor(7, 0.9), 1000, 200) === null);
  // 中间抖动一下：掉出第一名就要重新计时
  s.feed(probsFor(8, 0.9), 1100, 200);
  s.feed(probsFor(7, 0.9), 1200, 200);
  check('抖动后重新计时', s.feed(probsFor(7, 0.9), 1400, 200) === null);
  const r = s.feed(probsFor(7, 0.9), 1600, 200);
  check('稳定 400ms 后判赢', r !== null && r.outcome === 'win', JSON.stringify(r));
  check('赢得有分数', r !== null && r.score > 0, `score=${r?.score}`);
  check('状态变 won', s.state === 'won');
}

{
  // 概率排第一但没到阈值（0.42）——这才是「模型说了个别的词」的真实形态：
  // 148 类均分是 0.0068，随手一笔常在 0.15~0.3 之间，不能算过。
  const s = new Session('challenge', 3, N);
  s.start(0);
  let r = null;
  for (let t = 0; t <= 800; t += 40) r = r ?? s.feed(probsFor(3, 0.3), t, 200);
  check('未达阈值不算过', r === null, JSON.stringify(r));
}

{
  const s = new Session('challenge', 5, N);
  s.start(0);
  const r = s.tick(MODES.challenge.durationMs + 1);
  check('超时判输', r !== null && r.outcome === 'lose', JSON.stringify(r));
  check('输局零分', r !== null && r.score === 0, `score=${r?.score}`);
}

// ---------------------------------------------------------------- 盲区模式
console.log('\n盲区模式');
{
  const s = new Session('blind', 11, N);
  s.start(0);
  // EMA 单帧只取 42%，所以第一帧平滑后是 0.42*0.9+0.58*0.007 ≈ 0.38，还够不到
  // 0.42 的阈值 —— 真人画对一笔也不可能一帧就被看穿，这是对的。
  check('首帧不移交终局', s.feed(probsFor(11, 0.9), 0, 500) === null);
  s.feed(probsFor(11, 0.95), 100, 500);
  const r = s.feed(probsFor(11, 0.95), 300, 500);
  check('被看穿即判输', r !== null && r.outcome === 'lose', JSON.stringify(r));
}

{
  const s = new Session('blind', 11, N);
  s.start(0);
  // 撑满 40 秒，墨量 300（远超 minInk 90）。先喂一帧让状态机记下墨量——
  // 只推进 tick 不喂推理，ink 还是 0，会被判成「墨太少不算」。
  s.feed(probsFor(99, 0.9), 0, 300);
  const r = s.tick(MODES.blind.durationMs + 1);
  check('撑满 40s 判赢 —— 这就是「猜不中玩家赢」', r !== null && r.outcome === 'win', JSON.stringify(r));
  check('墨量决定得分', r !== null && r.score >= 600, `score=${r?.score}`);
}

{
  const s = new Session('blind', 11, N);
  s.start(0);
  // 墨量不足时不该给赢：喂一次极低墨量的推理，让它记下 ink
  s.feed(probsFor(99, 0.9), 10, 20);
  const r = s.tick(MODES.blind.durationMs + 1);
  check('墨量不足判平局，不算赢', r !== null && r.outcome === 'draw', JSON.stringify(r));
}

{
  // 危险信号：模型进了前三但还没说出口 —— 盲区模式里唯一的心跳加速来源。
  // EMA 单帧只取 42%，所以原始 p 得给到 0.4 才能让平滑值越过 0.12 这条线，
  // 同时又低于 0.42 的判定阈值（不能让它真的被看穿）。
  const s = new Session('blind', 2, N);
  s.start(0);
  s.feed(probsFor(2, 0.4), 0, 300);
  const snap = s.snapshot(500);
  check('进前三触发危险态', snap.danger === true, JSON.stringify({ rank: snap.rank, c: snap.closeness }));
  check('危险态下排名可见', snap.rank === 0, `rank=${snap.rank}`);
}

// ---------------------------------------------------------------- 自由模式
console.log('\n自由模式');
{
  const s = new Session('free', -1, N);
  // 自由模式一进来就是 playing，从未调用 start()，startedAt 仍是 0。
  // 这里钉住「别把 1970 年至今的毫秒数当用时显示出来」。
  const snap = s.snapshot(Date.now());
  check('自由模式用时归零（不是 5 亿秒）', snap.elapsedMs === 0, `elapsedMs=${snap.elapsedMs}`);
  check('自由模式无排名', snap.rank === -1);
  check('自由模式永不结算', s.feed(probsFor(1, 0.99), 1000, 400) === null);
  check('自由模式无倒计时', snap.remainMs === 0);
}

// ---------------------------------------------------------------- 计分与连击
console.log('\n计分');
{
  const s1 = new Session('challenge', 0, N, 0);
  s1.start(0);
  const plain = (() => {
    let r = null;
    for (let t = 0; t <= 600; t += 50) {
      const x = s1.feed(probsFor(0, 0.9), t, 300);
      if (x) r = x;
    }
    return r;
  })();

  const s2 = new Session('challenge', 0, N, 4);
  s2.start(0);
  const boosted = (() => {
    let r = null;
    for (let t = 0; t <= 600; t += 50) {
      const x = s2.feed(probsFor(0, 0.9), t, 300);
      if (x) r = x;
    }
    return r;
  })();

  check(
    '连击 4 层的分数高于无连击',
    boosted !== null && plain !== null && boosted.score > plain.score,
    `plain=${plain?.score} boosted=${boosted?.score}`,
  );
  // 4 层加成 = 1 + 4*0.15 = 1.6
  const ratio = boosted && plain ? boosted.score / plain.score : 0;
  check('加成幅度符合 1+0.15/层', ratio > 1.5 && ratio < 1.7, `ratio=${ratio.toFixed(3)}`);
}

// ---------------------------------------------------------------- 提示
console.log('\n提示');
{
  const s = new Session('challenge', 4, N);
  s.start(0);
  const h = s.useHint();
  check('提示有文案', typeof h === 'string' && h.length > 0, h);
  check('提示消耗计入 hintsUsed', s.hintsUsed === 1, `hintsUsed=${s.hintsUsed}`);
}

console.log('');
if (failures === 0) {
  console.log('玩法状态机：全部通过');
} else {
  console.log(`玩法状态机：${failures} 项未通过`);
  process.exit(1);
}
