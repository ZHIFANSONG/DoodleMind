// 造一份「第 1 关已全过」的进度，验证首页会切到第 2 关并显示星数
export default async (page) => {
  await page.evaluate(() => {
    localStorage.setItem(
      'doodlemind.progress.v1',
      JSON.stringify({ stages: { 0: { slots: [3, 3, 2, 3], bestMs: 41000 } } }),
    );
  });
  await page.evaluate(() => location.reload());
  await page.wait(9000);
  const text = await page.evaluate(() =>
    document.body.innerText.split('\n').filter((s) => s.trim()).slice(0, 16),
  );
  console.log('首页', JSON.stringify(text));
};
