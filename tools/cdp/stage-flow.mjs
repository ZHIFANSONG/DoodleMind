// 过关之后：确认进度落盘、确认「下一题」会走到下一格
export default async (page) => {
  const cx = 209;
  const cy = 376;
  const R = 92;

  const dump = async (tag) => {
    const info = await page.evaluate(() => ({
      prog: localStorage.getItem('doodlemind.progress.v1'),
      url: location.pathname + location.search,
      head: document.body.innerText.split('\n').filter((s) => s.trim()).slice(0, 3).join(' | '),
    }));
    console.log(tag, JSON.stringify(info));
  };

  // 圆 + 八道光芒
  await page.down(cx + R, cy);
  for (let i = 1; i <= 40; i++) {
    const a = (i / 40) * Math.PI * 2;
    await page.move(cx + R * Math.cos(a), cy + R * Math.sin(a));
  }
  await page.up();
  await page.wait(250);
  for (let k = 0; k < 8; k++) {
    const a = (k / 8) * Math.PI * 2;
    const x0 = cx + (R + 12) * Math.cos(a);
    const y0 = cy + (R + 12) * Math.sin(a);
    const x1 = cx + (R + 34) * Math.cos(a);
    const y1 = cy + (R + 34) * Math.sin(a);
    await page.down(x0, y0);
    await page.move((x0 + x1) / 2, (y0 + y1) / 2);
    await page.move(x1, y1);
    await page.up();
    await page.wait(80);
  }
  await page.wait(2500);
  await dump('过关后');

  // 用 DOM 量出「下一题」按钮的真实中心，避免手算坐标打空
  const spot = await page.evaluate(() => {
    const els = [...document.querySelectorAll('div')].filter((e) => e.textContent?.trim() === '下一题');
    const el = els[els.length - 1];
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2) };
  });
  console.log('按钮位置', JSON.stringify(spot));
  if (spot) await page.click(spot.x, spot.y);
  await page.wait(2600);
  await dump('点下一题后');
};
