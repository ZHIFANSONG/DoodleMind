// 边画边判的端到端验证：不松手，只在中途读候选栏。
// 期望：画到一半时 pct 与候选词已经在动（而不是等抬手才从 0 跳到 90%）。
export default async (page) => {
  const read = async (tag) =>
    page.evaluate(() => {
      const t = document.body.innerText.split('\n').filter((s) => s.trim());
      const idx = t.findIndex((s) => s.includes('它有多确定'));
      const pct = t[idx + 1] ?? '?';
      const note = t[idx + 2] ?? '';
      const ms = t.find((s) => /^\d+ms$/.test(s)) ?? '—';
      return { pct, note, ms };
    });

  console.log('落笔前', JSON.stringify(await read()));

  const cx = 215, cy = 470, r = 105;
  // 手指按住不放，绕一个圆，中途取样 4 次
  await page.down(cx + r, cy);
  for (let step = 1; step <= 24; step++) {
    const a = (step / 24) * Math.PI * 2;
    await page.move(cx + r * Math.cos(a), cy + r * Math.sin(a));
    if (step === 6 || step === 12 || step === 18 || step === 24) {
      await page.wait(220);
      console.log(`画到 ${Math.round((step / 24) * 100)}%（未松手）`, JSON.stringify(await read()));
    }
  }
  await page.up();
  await page.wait(300);
  console.log('松手后  ', JSON.stringify(await read()));
};
