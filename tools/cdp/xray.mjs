// X-Ray：画一只猫，开 X 光，看热区落在哪儿；再补一笔对比
export default async (page) => {
  const cx = 209;
  const cy = 400;
  const R = 92;

  const read = async (tag) => {
    const t = await page.evaluate(() => {
      const lines = document.body.innerText.split('\n').filter((s) => s.trim());
      return {
        head: lines.slice(0, 3).join(' | '),
        guesses: lines.filter((s) => /%$/.test(s)).slice(0, 4).join(' '),
        ms: lines.find((s) => /^\d+ms/.test(s)) ?? '—',
      };
    });
    console.log(tag, JSON.stringify(t));
  };

  const arc = async (cxx, cyy, r, from, to, n = 24) => {
    await page.down(cxx + r * Math.cos(from), cyy + r * Math.sin(from));
    for (let i = 1; i <= n; i++) {
      const a = from + ((to - from) * i) / n;
      await page.move(cxx + r * Math.cos(a), cyy + r * Math.sin(a));
    }
    await page.up();
    await page.wait(120);
  };

  // 猫头（圆）+ 两只耳朵
  await arc(cx, cy, R, 0, Math.PI * 2, 36);
  await page.down(cx - 70, cy - 62);
  await page.move(cx - 52, cy - 108);
  await page.move(cx - 26, cy - 74);
  await page.up();
  await page.wait(120);
  await page.down(cx + 70, cy - 62);
  await page.move(cx + 52, cy - 108);
  await page.move(cx + 26, cy - 74);
  await page.up();
  await page.wait(700);
  await read('开X光前');

  // 点「X 光」
  const btn = await page.evaluate(() => {
    const els = [...document.querySelectorAll('div')].filter((d) => d.textContent === 'X 光');
    const el = els[els.length - 1];
    const b = el.getBoundingClientRect();
    return { x: b.x + b.width / 2, y: b.y + b.height / 2 };
  });
  console.log('X光按钮', JSON.stringify(btn));
  await page.click(btn.x, btn.y);
  await page.wait(400);

  // 补一笔（眼睛），触发下一次推理 + 热力图刷新
  await page.down(cx - 30, cy - 10);
  await page.move(cx - 22, cy - 10);
  await page.up();
  await page.wait(600);
  await read('开X光后');
};
