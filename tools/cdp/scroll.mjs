export default async (page) => {
  const info = await page.evaluate(() => {
    const all = [...document.querySelectorAll("*")];
    const sc = all.filter((e) => e.scrollHeight > e.clientHeight + 40);
    sc.forEach((e) => { e.scrollTop = 1080; });
    window.scrollTo(0, 1080);
    return sc.length;
  });
  await page.wait(900);
  console.log("scrolled", info);
};

