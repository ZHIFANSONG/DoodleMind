# DoodleMind

**在手机上离线跑一个手写 CNN 的你画我猜。**

画布上画一只猫，屏幕下方实时跳出一排候选词和概率——1~2 毫秒一次，全部在你手机里算完。没有网络请求，没有云端推理，没有账号。

模型 163 KB，148 个类别，推理引擎是一段手写的 TypeScript。

```
assets/model/doodlemind-v1.json   163 KB   int8 权重 + base64
src/lib/model.ts                  ~200 行  im2col 卷积 / maxpool / GAP / FC
src/lib/raster.ts                 32×32 栅格化（与训练侧逐位一致）
```

iOS 上用 Expo Go 就能跑，不需要 Xcode、不需要 Mac、不需要开发者账号。

---

## 它好玩在哪

普通的你画我猜，对手是人，你知道他在想什么。这里的对手是一个 4 层卷积网络，**它对你不透明**。

所以游戏变成了逆向工程：什么样的轮廓能让这个黑箱说出「长颈鹿」？为什么画了斑纹它反而更糊涂？为什么把脖子画长一点概率就跳了 30 个点？

底部那条候选栏就是这个黑箱的注意力实时投影。看着它从「铅笔」滑到「长颈鹿」，是这类小游戏最有意思的一刻。

判定不是「top-1 命中」，而是三个条件同时成立：

1. 目标类概率经 EMA 平滑后 ≥ 0.42
2. 目标类在平滑后的排名里排第 1
3. 上面两条连续保持 380 ms

只用 top-1 会有一个很糟的体验：同一张图多画一笔，排名就翻转，玩家会觉得「明明画对了却不给过」。平滑 + 保持时长把这种抖动滤掉了，代价是判定感觉慢了半拍——这个取舍值得。

---

## 为什么不用 TensorFlow.js

一个 163 KB 的固定拓扑模型，用 tfjs 意味着拉进 ~1.5 MB 的运行时、一层 WebGL/WebGPU 后端适配、还有 JIT 之外的算子调度开销。而我们需要的东西只有四种算子。

手写的版本：

- **零依赖**，`src/lib/model.ts` 只 import 了 `Float32Array`
- **~7M 次乘加**，iPhone 13 上单次 3~8 ms（比一帧 16 ms 还短，所以可以在手指移动时同步跑）
- 复用两个 `Float32Array` 缓冲，稳态运行时**零分配**，不触发 GC
- 首次 forward 空跑一次做 V8 预热，避免第一笔卡顿

代价是这段代码只认我们自己的二进制格式（见下），换模型要重新导。对一个自包含的小工具，这个代价是划算的。

## 权重是 int8，激活是 float32

卷积权重按输出通道做对称量化（`w_scale = max|w| / 127`），4 个卷积块共 163 KB。**但激活不量化**——解量化后全用 `Float32Array` 算。

原因很直接：JS 里 int8 整数运算并不比 float 快（没有 SIMD，反而多一层拆包/重打包），而激活量化会在 4 段卷积里累积误差，实测掉好几个点准确率。权重量化省的是**包体积**和**解析时间**，这才是我们真正需要省的东西。

BatchNorm 在导出时被折进卷积（`k = γ/√(σ²+ε)`，`w' = w·k`，`b' = β − μ·k`），端上不用管 BN。

---

## 数据与训练

数据来自 [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset)（Google，CC BY 4.0）。用的是 `simplified` 格式——**原始笔画序列**，不是渲染好的位图。

这一点很重要：我们自己栅格化，所以端上和训练侧的输入口径能逐位对齐。`tools/raster.py` 是权威实现，`src/lib/raster.ts` 是它的逐行移植，两边都是：

```
包围盒 → 等比缩放到 32−2×3 → 居中 → 沿线段画半径 1 的圆头笔
覆盖率 cov = clip(radius + 0.5 − dist, 0, 1)，同像素取 max（抗锯齿）
```

如果改用官方 `numpy_bitmap`（28×28、按包围盒居中渲染），端上就得复现它的渲染管线，而且 28×28 上采样到 32×32 会糊。自己栅格化反而更简单。

```bash
python tools/fetch_data.py --per-class 4500    # 拉数据、栅格化 → tools/out/doodles.npz
python tools/train.py                          # → tools/out/doodlenet.pt
python tools/export_model.py                   # → assets/model/doodlemind-v1.{bin,json}
node  tools/check-engine.mjs                   # 跨语言一致性校验
```

训练配方：AdamW + OneCycle、label smoothing 0.08、30% 概率 mixup、仿射增强（旋转 ±13°、缩放 0.82–1.14、gamma 抖动）——都在 CPU 上跑，20 核大约 40 分钟。

### 跨语言校验是必须的

PyTorch 训出来的模型，和 JS 里手写的推理，数学上应该完全一样。但「应该」不算数。

`tools/export_model.py` 会导出 120 张测试图和 PyTorch 的参考预测；`tools/check-engine.mjs` 拿这些图喂给 TS 引擎，逐张比对 top-1，**一致率低于 99.5% 就直接 exit 1**。

这道闸门抓到过真问题：卷积的 `ky/kx` 索引顺序、有效区间裁剪的边界、pool 在奇数尺寸上的取整。

---

## 跑起来

```bash
npm install
npx expo start     # 手机装 Expo Go，扫码
```

Windows 上开发完全够用——写代码、热重载、真机预览全都不需要 Mac。

```bash
npx eslint src         # lint
npx tsc --noEmit       # 类型检查
```

> `npx expo lint` 在 SDK 58 + ESLint 9 下会误报版本检查失败，直接调 `npx eslint` 即可。

---

## 目录

```
src/
  app/
    index.tsx          游戏主界面（题目 / 画布 / 候选栏 / 结算）
    gallery.tsx        画册：存的是笔画矢量，不是截图
    about.tsx          技术说明 + 能真画的调试台
    _layout.tsx        锁定浅色主题
  components/
    drawing-canvas.tsx PanResponder 画布 + Catmull-Rom 平滑
    app-tabs.tsx       三个 tab
    splash-overlay.tsx 启动过渡
  lib/
    model.ts           手写 CNN 推理
    raster.ts          32×32 栅格化
    game.ts            局状态机、EMA 判定、计分
    history.ts         AsyncStorage 持久化
tools/
  fetch_data.py        数据管线（流式读 ndjson，读够就断）
  raster.py            栅格化权威实现
  train.py             PyTorch 训练
  export_model.py      折 BN / int8 量化 / 二进制打包
  check-engine.mjs     跨语言一致性闸门
```

画册里存的是**笔画矢量**（`[[x,y],...]`，8~20 KB）而不是渲染好的截图（1080p PNG 约 400 KB）。重绘时按当前屏幕宽度重新算路径，所以缩略图在任何设备上都清晰。

---

## 模型格式（v2）

全小端：

```
"DDMN"                    4B   magic
<B nBlocks, H nClasses, H>…   头
  每块：<H cin, H cout>  f32 wScale[cout]  f32 bias[cout]  int8 w[cout*cin*9]
<H width>                 f32 headW[nClasses*width]  f32 headB[nClasses]
```

权重行主序 `(cout, cin, ky, kx)`。端上解量化成 `Float32Array` 常驻，163 KB → 650 KB 内存。

---

## 已知限制

- **148 类**，是从 QuickDraw 345 类里挑的「画得出来、认得出来」的词。冷门词（比如「潜望镜」）容易和近义词混。
- 模型只吃**轮廓**，不吃颜色。6 色调色板纯粹是给人看的，不影响推理。
- 32×32 意味着细节会丢。画大、线条少、单一主体，成功率最高。
- 类别固定，不能自己加词——加词得重训 + 重导，`tools/` 里全套脚本都在，改 `tools/categories.py` 就行。

## 致谢

- [QuickDraw Dataset](https://github.com/googlecreativelab/quickdraw-dataset) — Google Creative Lab，CC BY 4.0
- [Expo](https://expo.dev) — 让 Windows 也能开发 iOS App
- 参考 [MiniWatts](https://github.com/mickeu/miniwatts) 的定位：单一用途、自签、不上架、认真做的小工具

## License

MIT
