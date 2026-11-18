// 验证「按住不放时墨是否跟着手指走」：画一条不回头的线，松手前截一张图。
// 脚本故意**不抬手**——cdp.mjs 在脚本结束后才截图，所以截到的是「手指还按着」
// 的那一帧，正好能看出 live 笔画是否已经画到最后一格。
export default async (page) => {
  const read = async () =>
    page.evaluate(() => {
      const t = document.body.innerText.split('\n').filter((s) => s.trim());
      const i = t.findIndex((s) => s.includes('它有多确定'));
      return {
        pct: t[i + 1] ?? '?',
        note: t[i + 2] ?? '',
        ms: t.find((s) => /^\d+ms$/.test(s)) ?? '—',
        counts: t.find((s) => /笔 · \d+s · 墨/.test(s)) ?? '—',
        paths: document.querySelectorAll('svg path').length,
      };
    });

  console.log('落笔前', JSON.stringify(await read()));

  // 画一个逗号：从左上斜下，再向右弯
  const pts = [];
  for (let i = 0; i <= 30; i++) {
    const a = (i / 30) * Math.PI * 1.5;
    pts.push([215 + 90 * Math.cos(a + 2.2), 430 + 90 * Math.sin(a + 2.2)]);
  }
  await page.down(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) {
    await page.move(pts[i][0], pts[i][1]);
    if (i === 8 || i === 16 || i === 24 || i === 30) {
      await page.wait(160);
      console.log(`第 ${i}/30 点（手指未松）`, JSON.stringify(await read()));
    }
  }
  await page.wait(400);
  console.log('截图前一刻（仍未松手）', JSON.stringify(await read()));
};
