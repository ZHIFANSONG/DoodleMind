/**
 * 用 Edge 的 CDP 端口做「能点能画还能截图」的预览器。
 *
 * 为什么需要这个：ego-browser 在本机出不了像素（Page.captureScreenshot 永远超时），
 * 而 Edge 的 --headless=new --screenshot 只截得到闪屏。但 Edge 有真窗口时
 * CDP 是完全可用的——既能派发鼠标事件，也能 Page.captureScreenshot 拿到真实画面。
 *
 * 用法：
 *   npx expo start --web --port 8081        # 先在另一个终端起 dev server
 *   node tools/cdp/cdp.mjs <route> <out.png> [scriptFile]
 *   scriptFile 里可以写 export default async (page) => {...}，page 是薄封装。
 *
 * Edge 的路径写死成 Windows 默认安装位置，换机器要改 EDGE 常量。
 */
import { spawn } from 'node:child_process';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9333;

const route = process.argv[2] ?? '/';
const out = process.argv[3] ?? 'shot.png';
const scriptPath = process.argv[4];

const profile = mkdtempSync(join(tmpdir(), 'edge-cdp-'));

const child = spawn(
  EDGE,
  [
    `--app=http://localhost:8081${route}`,
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${profile}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--window-size=430,932',
    '--window-position=60,40',
    '--force-device-scale-factor=2',
    '--disable-features=Translate,TranslateUI,msEdgeTranslate,EdgeCollections,msEdgeSplitScreen,msFeedback',
  ],
  { stdio: 'ignore', detached: false },
);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function findTarget() {
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      const list = await res.json();
      const page = list.find((t) => t.type === 'page' && t.url.includes('localhost:8081'));
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch {
      /* 还没起来 */
    }
    await sleep(500);
  }
  throw new Error('CDP 端口没起来');
}

const wsUrl = await findTarget();
const ws = new WebSocket(wsUrl);
await new Promise((res, rej) => {
  ws.onopen = res;
  ws.onerror = rej;
});

let seq = 0;
const pending = new Map();
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.id && pending.has(msg.id)) {
    const { res, rej } = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result);
  }
};

function send(method, params = {}) {
  const id = ++seq;
  return new Promise((res, rej) => {
    pending.set(id, { res, rej });
    ws.send(JSON.stringify({ id, method, params }));
    setTimeout(() => {
      if (pending.delete(id)) rej(new Error(`timeout: ${method}`));
    }, 30000);
  });
}

const page = {
  _x: 0,
  _y: 0,
  async evaluate(fn, ...args) {
    const expr = `(${fn.toString()})(${args.map((a) => JSON.stringify(a)).join(',')})`;
    const r = await send('Runtime.evaluate', {
      expression: expr,
      awaitPromise: true,
      returnByValue: true,
    });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text + ' ' + (r.exceptionDetails.exception?.description ?? ''));
    return r.result.value;
  },
  async move(x, y) {
    page._x = x;
    page._y = y;
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y, button: 'none', buttons: 0 });
  },
  async down(x, y) {
    page._x = x;
    page._y = y;
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', buttons: 1, clickCount: 1 });
  },
  async up(x, y) {
    // 参数可省：CDP 的 mouseReleased 里 x/y 是必填的，漏传就报
    // "Failed to deserialize params.x - BINDINGS: mandatory field missing"。
    await send('Input.dispatchMouseEvent', {
      type: 'mouseReleased',
      x: x ?? page._x,
      y: y ?? page._y,
      button: 'left',
      buttons: 0,
      clickCount: 1,
    });
  },
  async click(x, y) {
    await page.move(x, y);
    await sleep(60);
    await page.down(x, y);
    await sleep(60);
    await page.up(x, y);
  },
  wait: sleep,
};

await send('Runtime.enable');
await send('Page.enable');
// 等页面真正渲染出来（模型要加载）
await sleep(9000);

if (scriptPath) {
  const mod = await import(pathToFileURL(scriptPath).href);
  await mod.default(page);
}

const shot = await send('Page.captureScreenshot', { format: 'png' });
writeFileSync(out, Buffer.from(shot.data, 'base64'));
console.log(`saved ${out}`);

ws.close();
child.kill();
await sleep(600);
process.exit(0);
