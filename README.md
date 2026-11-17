# DoodleMind

**在手机上离线跑一个手写 CNN 的你画我猜。**

画布上画一只猫，屏幕下方实时跳出一排候选词和概率——1~2 毫秒一次，全部在你手机里算完。没有网络请求，没有云端推理，没有账号。

模型 310 KB、307K 个参数、148 个类别，推理引擎是一段手写的 TypeScript。

```
assets/model/doodlemind-v4.json   421 KB   int8 权重 + base64（打包进 App 的就是它）
assets/model/doodlemind-v4.bin    310 KB   同一份权重的紧凑二进制（调试用）
assets/images/icon.png             62 KB   用 App 自己的笔刷画的
src/lib/model.ts                  ~330 行  3×3 卷积 / depthwise / 1×1 / maxpool / GAP / FC
src/lib/raster.ts                 32×32 栅格化（与训练侧逐位一致）
```

| 开始 | 挑战 | 关于 |
| --- | --- | --- |
| ![首页](docs/shots/home.png) | ![挑战](docs/shots/v4-play.png) | ![关于](docs/shots/about.png) |

（截图来自 web 预览，所以顶部那条深色胶囊是 `NativeTabs` 的 web 实现——它在 iOS 上是系统原生的底部标签栏。）

## 三种玩法

| 玩法 | 谁出题 | 你怎么赢 |
|---|---|---|
| **挑战** | App 给词，60 秒 | 让它说出这个词（概率 ≥ 0.42 且排名第一，保持 380 ms） |
| **盲区** | App 给词，40 秒 | 你**故意画得让它认不出**——被看穿就输，撑满 40 秒就赢 |
| **自由** | 无题 | 一直画，看它怎么想 |

盲区模式是反向的：你和一个神经网络对抗，而它的判决标准写在明面上。墨量少于阈值不算赢（不然空着画布就能过关），墨越多得分越高。

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

一个固定拓扑的小模型，用 tfjs 意味着拉进 ~1.5 MB 的运行时、一层 WebGL/WebGPU 后端适配、还有 JIT 之外的算子调度开销。而我们需要的东西只有四种算子。

手写的版本：

- **零依赖**，`src/lib/model.ts` 只 import 了 `Float32Array`
- **约 9M 次乘加**，实测单次 8.5 ms（比一帧 16 ms 还短，所以可以在手指移动时同步跑）
- 复用两个 `Float32Array` 缓冲，稳态运行时**零分配**，不触发 GC
- 首次 forward 空跑一次做预热，避免第一笔卡顿

代价是这段代码只认我们自己的二进制格式（见下），换模型要重新导。对一个自包含的小工具，这个代价是划算的。

## 为什么是可分离卷积

这一节值得单独写，因为「模型要更聪明」和「要能在手机上实时跑」看起来矛盾，但解法不是取中间值，而是换结构。

先量清楚瓶颈。`tools/bench_layers.mjs` 把一次推理的乘加数按层摊开：

| 层 | 形状 | 乘加 | 占比 |
| --- | --- | --- | --- |
| block1 | 32×32, 1→32 | 0.29 M | 2.9% |
| block2 | 16×16, 32→64 | 4.72 M | **45.6%** |
| block3 | 8×8, 64→96 | 3.54 M | 34.2% |
| block4 | 4×4, 96→128 | 1.77 M | 17.1% |
| head | 128→148 | 0.02 M | 0.2% |

再对比实测时间：10.34 M 次乘加跑 11.38 ms，**有效算力 0.91 GMAC/s**。时间和乘加数几乎是线性关系——说明瓶颈不在循环写法，而在乘加数本身。

先确认循环没得优化了，两个反直觉的实验：

- 把 9-tap 展开改成「按 tap 连续累加」：**慢 0.75 倍**
- 改成「行缓冲累加」（一行输出只写一次）：**慢 0.82 倍**

现有的 9-tap 展开已经贴着 V8 在这套内存布局上的上限。这条路走到头了。

于是换结构。同形状下（16×16, 32→64）实测：

| 实现 | 乘加 | 时间 |
| --- | --- | --- |
| 普通 3×3 | 4.72 M | 4.108 ms |
| depthwise 3×3 + pointwise 1×1 | 0.60 M | 0.594 ms |

**实测加速 6.92 倍**（理论 7.89 倍），可分离卷积几乎吃满了理论收益。省下来的预算全部换成通道数：从 24/48/96/128 加到 48/160/320/512，参数量 182K → 307K，而推理时间从 11.38 ms 降到 8.18 ms。

准确率也没有让步——同一份 4500 张/类的数据、70 轮训练：

| 结构 | 参数量 | 验证集 top-1 | top-5 | 端上单次推理 |
| --- | --- | --- | --- | --- |
| 普通 3×3（24/48/96/128） | 182K | 0.8138 | 0.9609 | 11.38 ms |
| 可分离（48/160/320/512） | 307K | **0.8180** | **0.9635** | **8.18 ms** |

也就是说：**更准、更宽、更快，同时**。代价是实现里多两种算子（depthwise 和 1×1），`src/lib/model.ts` 从 3 个函数变成 5 个。

## 权重是 int8，激活是 float32

卷积权重按输出通道做对称量化（`w_scale = max|w| / 127`），fp32 的几 MB 压到这个体积（见上方规格表与「关于」页）。**但激活不量化**——解量化后全用 `Float32Array` 算。

原因很直接：JS 里 int8 整数运算并不比 float 快（没有 SIMD，反而多一层拆包/重打包），而激活量化会在几段卷积里累积误差，实测掉好几个点准确率。权重量化省的是**包体积**和**解析时间**，这才是我们真正需要省的东西。

分类头（`128→148` 的那层全连接）也量化成 int8。它占了整包三成的体积，而 GAP 的输出非负、量级集中，每类的动态范围差异正好被 per-class scale 吃掉——实测只掉 0.2 个点。

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
python tools/train.py --arch sep --width 48,160,320,512 \
                      --epochs 70 --batch 256 --per-class 4500 --device cuda \
                      --out doodlenet_sep.pt    # → tools/out/doodlenet_sep.pt
python tools/export_model.py --ckpt doodlenet_sep.pt --name doodlemind-v4
node  tools/check-raster.mjs                   # 第一道：栅格化
node  tools/check-engine.mjs                   # 第二道：跨语言推理一致性
node  tools/check-game.mjs                     # 第三道：玩法状态机
```

`check-engine.mjs` 不用传模型名：它会从 `src/hooks/use-doodle-ai.ts` 里读出 App 实际 `require` 的那个文件。写死常量意味着换模型时要记得改两处，而「忘了改」的后果不是报错，是 CI 兴高采烈地校验一个早已淘汰的旧模型、然后放行一个没验过的新模型。

参考向量也按模型名分文件（`tools/vectors/test_vectors_<name>.json`），缺了直接报错。这一点是被真事故教出来的：早期用固定文件名，导出探针模型时把主力模型的向量覆盖了，于是「一致率 67%」看起来像引擎写错，实际是标尺拿错了。

训练配方：AdamW + OneCycle、label smoothing 0.08、30% 概率 mixup、仿射增强（旋转 ±13°、缩放 0.82–1.14、gamma 抖动）、以及为「边画边判」加的 cutout（见下）。

`tools/train.py` 认 `--device auto|cpu|cuda`。这个网只有 30 万参数，看起来「这么小还要 GPU？」——但它一轮要过 63 万张图，瓶颈在吞吐不在显存：

| 设备 | 每轮（sep, 4500/类） |
| --- | --- |
| CPU（20 核） | 约 4–5 分钟 |
| RTX 3060 12G | **69 s** |

想用 GPU 得装 CUDA 版 torch（默认从 PyPI 装的是 `+cpu` 构建，`torch.cuda.is_available()` 直接是 `False`）：

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 --upgrade "torch==2.14.1+cu126"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python tools/train.py --device cuda
```

`--device auto` 是默认值：有卡就用卡，没卡回落到 CPU，不会因为「装了 CUDA 版但驱动不认」而中途炸掉。存档时权重一律 `.detach().cpu()`——导出脚本在纯 CPU 上跑，存 CUDA 张量会让它必须记得传 `map_location`，漏一次就是一堆难查的报错。

### 第一道闸门：栅格化

`tools/check-engine.mjs` 喂的是**已经栅格化好的 32×32 灰度图**，它只能验证推理，验证不了「笔画 → 图」这一步。而栅格化才是训练侧和端上唯一共享的几何口径——这一步偏一个像素，模型再准也白搭。

所以先有这条独立的校验：

```bash
node tools/check-raster.mjs
```

它做两件事：

1. **逐像素比对**。`tools/make_raster_vectors.py` 从 7 个类各取样例，外加 3 组人工构造的偏心/极小样本（贴着画布边角的线、两像素的点），用 Python 权威实现算出 32×32 灰度图存进 JSON，TS 移植版必须对得上（允许 ≤1 级灰度差，浮点舍入）。
2. **平移不变性断言**。把同一组笔画整体平移 +137/+91 像素后重新栅格化，结果必须**完全一致**。这一条不依赖任何参考数据，只依赖几何本身——「画在左上角和画在右下角应该得到同一张图」。

第二条抓到的 bug 比推理那边更隐蔽。`rasterizeInto` 里投影时只写了 `ax = xs[i] * scale + offX`，**忘了减包围盒原点**。训练侧看不出来，因为 QuickDraw simplified 数据有 88.6% 的样本 `min_x` 恰好是 0（4790 张猫里 4245 张），减不减零都一样；而用户的手指可以落在画布任何位置。表现是：一条贯穿画布的**水平直线，墨量 0**，一个**单点，墨量 0**——它们被画到了 32×32 外面，模型收到一张全黑图。

修完实测：一条直线 55、一个圆 163、一只猫 187（这三组数就是墨量量纲的参照）。

### 第二道闸门：跨语言推理一致性

PyTorch 训出来的模型，和 JS 里手写的推理，数学上应该完全一样。但「应该」不算数。

这里有个方法论上的坑值得写下来。第一版的做法是：导出 PyTorch **原始 fp32** 权重下的预测当参考，拿 TS 跑 int8 权重的结果去比。结果一致率只有 57.5%，而这个数字什么也说明不了——它同时混进了两个变量：int8 量化本身的误差，和我实现的 bug。阈值定在哪里都是错的。

改成**参考端也用解量化后的同一套 int8 权重**跑 torch，两台机器的输入才真正相同。这时剩下的任何偏差都只可能是实现问题。

`tools/export_model.py` 导出 120 张测试图、参考 top-1，以及**每张图 top-5 的概率值**；`tools/check-engine.mjs` 的闸门因此变成两条：

- top-5 概率的**最大绝对偏差 < 1e-3**
- top-1 一致率 ≥ 99.5%

第一条才是真正的正确性判据。只看 argmax 是不够的——概率差 1e-6 和差 1e-1 在 top-1 上可能完全看不出来，但后者说明算错了。

这道闸门抓到过真问题：卷积 `ky/kx` 的索引顺序、有效区间裁剪的边界写反（`y1 = din > 0 ? h : h + din` 应该是 `h - din`——越界读不崩溃，只是模型悄悄变笨）、pool 在奇数尺寸上的取整。

实测 120/120 一致，概率最大偏差 `9.7e-7`。

### 第三道闸门：玩法状态机

前两道管的是「看起来对不对」，这道管的是「游戏成不成立」。

三种玩法唯一的差别就是判定规则：挑战模式是「模型说出目标词就赢」，盲区模式恰好相反——「模型整局都没说出目标词才赢」。盲区的超时到底算赢、算平、还是算输？墨量不够算不算？连击加成到底乘上去了没有？这些靠人肉在真机上试，一次要等 40 秒，试十次废掉一个下午，而且容易记错。

判定逻辑没有 UI、不碰 React、纯函数式推进，可以完全脱离手机跑：

```bash
node tools/check-game.mjs
```

29 条断言分成五组——挑战、盲区、自由、计分、提示。它抓到过一个真 bug：`holdStartedAt` 用 `0` 当「还没开始保持」的哨兵值，而 `0` 是合法时间戳，测试里正好从 `now = 0` 起算，于是每一帧都被判成「还没开始」，保持时长永远攒不满，判定静默失效。改成 `-1` 就好——这种 bug 在真机上表现为「明明画对了就是不给过」，几乎不可能靠手感定位。

它还顺带钉住了自由模式那个 `1794912810s`（自由模式从不调 `start()`，`startedAt` 停在 0，用时算出来是 1970 年至今的毫秒数）。

### 墨量的量纲

`ink` 一开始是灰度总和（0..261120）。这个数没有直觉，没人知道 47000 算多还是少，代码里也没法定阈值。

改成**等效满墨像素数**（总和 / 255，0..1024）之后就能一眼判断了。盲区模式的「耍赖线」就架在这个量纲上：墨量少于 90 不算赢（约等于画完一个完整轮廓），否则空着画布撑满 40 秒就能过关。

---

## 界面：能用系统控件的地方就不用自己画的

`src/components/native/` 是一层薄薄的「原生控件」，目录下两份实现：

- `kit.ios.tsx` —— `@expo/ui/swift-ui`，真正的 SwiftUI 视图
- `kit.tsx` —— web / Android 上的 RN 回退
- `kit-types.ts` —— 两份实现共享的 props 契约

`kit-types.ts` 存在的理由值得说：Metro 按平台只打包一份实现，**TypeScript 也只看得到一份**。props 类型写在某一侧，另一侧写错了 `tsc` 永远发现不了，只会在真机上少一个 prop、少一个回调，而且不报错。把类型提到第三份文件，两边都 import 它，才能让类型检查覆盖两份实现。

iOS 上具体用了什么：

| 位置 | 系统控件 | 自己画的版本差在哪 |
| --- | --- | --- |
| 底部导航 | `UITabBar`（`NativeTabs`） | 点击高亮 / VoiceOver / 动态字体 / 重复点回顶部，全都没有 |
| 大标题 | `UINavigationBar` 的 large title | 滚动时标题缩小吸顶、背景渐变毛玻璃、状态栏变色，自己拼的 View 做不到跟着转场动 |
| 置信度条 | SwiftUI `Gauge` + `linearCapacity` | 轨道、端点圆角、动画、VoiceOver 的「进度」语义；系统换样式时它跟着变 |
| 规格表 | SwiftUI `LabeledContent` | 左右两列，VoiceOver 读成「标签，值」而不是两个孤立字符串 |
| 画册空态 | `ContentUnavailableView`（iOS 17+） | 和「设置」「App Store」里同一个控件，间距字重由 UIKit 排 |
| 长按删除 | SwiftUI `Menu` / `ContextMenu`（`MenuView`） | 系统的按压反馈、破坏性操作的红色、触觉反馈、iPad 上的弹出方向 |
| 候选栏毛玻璃 | `GlassView`（iOS 26 Liquid Glass） | 旧系统退回 `UIBlurEffect`，web 退回带纸色蒙版的 View |

三个反复踩到的坑，都在 `kit.ios.tsx` 顶部注释里：

1. `@expo/ui` 的每个 swift-ui 组件底层是 `requireNativeView('ExpoUI', ...)`，**必须包在 `Host` 里**。挂在 Host 外面不会报错，只是什么都不显示。
2. `Host` 不给 `matchContents` 时默认铺满父容器——放在一列 View 中间会吃掉全部剩余高度，把下面的元素顶出屏幕。
3. SwiftUI 的 `VStack` 按子视图**理想宽度**排布，撑满要显式 `frame({ maxWidth: 10000 })`（没有 `.infinity` 常量可用）。

没有系统控件可用、或者系统控件会碍事的地方，才自己画：手绘画布（压感笔迹必须自己算）、候选栏的 chip 流（横向滚动 + 实时重排，系统没有对应控件）、玩法卡片（卖的是三套规则，不是列表项）。

### 一个性能坑

`play.tsx` 的计时器每 33 ms 触发整页重渲染。判读区里的置信度条在 iOS 上是真 SwiftUI `Gauge`，每次重渲染都要跨 RN → SwiftUI 边界序列化一遍 props——不做处理的话，一秒有 30 次无意义的跨边界调用，而百分比一秒才变一两次。

所以判读区抽成了 `src/components/play-meter.tsx`，`React.memo` 包住，并且 **props 全部压成原始值**：`pct` 传整数百分比（不是 0..1 的浮点，浮点几乎每帧都在动）、`secs` 传整秒、`ink` 传取整后的墨量。浅比较这才挡得住绝大多数帧。

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

## 边画边判：它在你还没画完的时候就开口了

难点不在推理速度，在**训练数据的分布**。QuickDraw 里全是画完的图，只喂这些，模型会养成「等笔画齐了再表态」的习惯：画到第三笔还是 12%，一松手跳到 90%。人会觉得它反应慢，其实它一直在等。

三件事一起改：

**1. 训练时把图抹掉一块。** `tools/train.py` 的 `cutout()`：

- 半平面抹除（概率 0.25）——随机把上/下/左/右一半置零，最像「只画了一半」
- 矩形挖洞（概率 0.45）——边长 18%~42% 的方块，随机位置

等于告诉它「只有这些证据也要给答案」。推理路径一点没变重，只是学会了不依赖完整轮廓。

**2. 节流间隔按实测耗时自适应。** 原来是固定 140 ms：慢机器上照样掉帧，快机器上候选栏半秒不跳一次。现在跑完一帧就把下一次排到 `max(60, cost × 2.2)`——快机多猜，慢机自己退让，下限 60 ms（再快人眼也分不出，只烧电）。

**3. 落笔那一刻就开始推理。** 原先 `onPoint` 里写着 `if (state !== 'playing') return`，而把状态推到 `playing` 的 `start()` 要等到抬手才调用——于是整条第一笔画得再久，候选栏都是空的。现在 `onPoint` 自己负责开表。

顺带修了一个同类问题：判读区的「N 笔」原来读 `strokes.length`，而 `setStrokes` 只在抬手时调用，所以整条笔画过程中它一直显示「还没落笔，它只能瞎猜」。改成 `liveCount`，跟推理一起更新。

顺带还有一个：卷积的循环顺序从「输入通道在外」改成「输出通道在外」。原来每个 tap 都要把 32×32×128 的输出平面（512 KB）各摸一遍，远超 CPU 缓存。Node 上 16.4 ms → 13.7 ms。

边画边判的效果可以用 `cdp.mjs` 的脚本量化：按住不放绕一个圆，在 25% / 50% / 75% / 100% 处读判读区。修之前各阶段读数是 `—`（从没推理过），修之后是 `11 / 9 / 8 / 9 ms`，排名一路在动——全程手指没松。

### 怎么证明它真的「边画边判」了

拿最终 top-1 去衡量是没用的：那个数字是在**画完的图**上算的，模型完全可能画完 81%、半张图 30%，体感就是「一直不说话，松手才跳」。

`tools/eval_partial.py` 按笔画总弧长把样本截断到 25% / 50% / 75% / 100%，再过**同一条栅格化管线**（直接 `from fetch_data import rasterize_np`，不另抄一份几何口径——口径一旦分家，两个脚本会给出互相矛盾的结论而且都不报错）。

指标也不止 top-1。25% 弧长时人自己也认不出画的是什么，top-1 必然极低，但那不代表体验差。玩家真正看的是候选栏里有没有对的词、对的词排第几——所以还统计**前 6 命中率**（候选栏正好 6 格）和 **MRR**。

1460 张未参与训练的原始样本，最终版（可分离卷积）与早期版（普通 3×3）的对照：

| 弧长 | top-1 | 前 6 命中 | MRR | 排名中位 |
|---|---|---|---|---|
| 25% | 0.049 vs 0.052 | 0.146 vs 0.148 | 0.110 vs 0.110 | 32 vs 37 |
| **50%** | **0.201 vs 0.178** | **0.545 vs 0.481** | **0.341 vs 0.311** | **4 vs 6** |
| **75%** | **0.583 vs 0.553** | **0.884 vs 0.865** | **0.709 vs 0.683** | 0 vs 0 |
| 100% | **0.838 vs 0.817** | 0.978 vs 0.975 | **0.897 vs 0.884** | 0 vs 0 |

（左=可分离 48/160/320/512，右=普通 3×3 24/48/96/128。粗体行是「边画边判」真正吃劲的区间。）

增益在**中段最明显**：50% 弧长时 top-1 +2.3 点、前 6 +6.4 点、MRR +0.030；75% 时 top-1 +3.0 点。两端几乎不变——刚起笔时凭什么都认不出，画完时本来就不缺证据。而且这不是拿实时性换来的：100% 那一列同样涨了 2.1 个点。

另一组对照（有 cutout 与无 cutout，同结构）也值得看一眼，它证明的是「训练数据分布」这件事：50% 弧长时 top-1 +6.1 点、前 6 +4.0 点，而 100% 几乎不动。两个改动叠加，才让「边画边判」从效果上成立，而不只是从代码上成立。

---

## 装到 iPhone 上：自己出 ipa

`.github/workflows/ios.yml` 在 `macos-15` 上跑，产出**未签名**的 `DoodleMind-unsigned.ipa`。Windows 上签不了名，也不需要——签名是安装那一步的事。

三种触发方式：

```bash
# 1. 推 tag，自动发 Release，直接下 ipa
git tag v1.0.0 && git push origin v1.0.0

# 2. 手动跑一次（GitHub 页面 Actions → iOS → Run workflow）
# 3. 开 PR，只验证能不能编过，不发 Release
```

构建流程：`npm ci` → `tsc` + `eslint` → **三道纯逻辑闸门先跑**（栅格化 / 跨语言一致性 / 状态机，模型没导出好就不该浪费 20 分钟编译）→ `expo prebuild` 生成 `ios/` → `pod install` → `xcodebuild` 关掉一切签名 → 打包成 ipa → 校验结构（有 `Payload/*.app`、无 `_CodeSignature`、包内不含 `/Users/runner` 这类构建机路径）→ 上传 artifact。

下载后任选一个自签工具装：

| 工具 | 说明 |
| --- | --- |
| [Sideloadly](https://sideloadly.io) | Windows 上直接拖 ipa 进去，填 Apple ID，7 天有效 |
| [AltStore](https://altstore.io) / [SideStore](https://sidestore.io) | 装一次，之后每 7 天自动续签（SideStore 只需每 7 天在同一 Wi-Fi 下开机一次） |
| Xcode | 有 Mac 的话，`Window → Devices` 拖进去 |

免费 Apple ID 签的 App **7 天过期**，重签即可，画册里存的记录不会丢（在 App 沙盒里）。

---

## 目录

```
src/
  app/
    _layout.tsx                根 Stack：主题、启动过渡、play 的转场
    play.tsx                   游戏页（题目 / 画布 / 候选栏 / 结算）
    (tabs)/
      _layout.tsx              NativeTabs：系统 UITabBar
      (home)/index.tsx         首页：三种玩法的规则卡
      (gallery)/gallery.tsx    画册：存的是笔画矢量，不是截图
      (about)/about.tsx        技术说明 + 能真画的调试台
  components/
    native/
      kit-types.ts             两份实现的共享 props 契约
      kit.ios.tsx              SwiftUI 真控件（Host + VStack/Gauge/LabeledContent）
      kit.tsx                  web / Android 回退
    drawing-canvas.tsx 手写触摸画布 + Catmull-Rom 平滑 + 压感变宽
    play-meter.tsx     判读区（memo，挡住计时器的高频重渲染）
    glass.tsx          Liquid Glass / UIBlurEffect / 纸色蒙版三级回退
    glyph.tsx          图标：iOS 走 SF Symbol，其它平台用 SVG 现画
    splash-overlay.tsx 启动过渡
  lib/
    model.ts           手写 CNN 推理
    raster.ts          32×32 栅格化
    modes.ts           三种玩法的规则与判定
    session.ts         一局的状态机
    game.ts            EMA 平滑、排序、计分、提示
    history.ts         AsyncStorage 持久化
tools/
  fetch_data.py        数据管线（流式读 ndjson，读够就断）
  raster.py            栅格化权威实现
  train.py             PyTorch 训练（--arch dense|sep、cutout 增强、--device cuda）
  export_model.py      折 BN / int8 量化 / 二进制打包 / 生成参考向量
  make_raster_vectors.py  导出栅格化参考样本
  eval_partial.py      按弧长截断样本，量化「边画边判」的中间态表现
  bench_layers.mjs     按层统计乘加数与占比，找性能预算在哪
  bench_sep.mjs        可分离卷积 vs 普通 3×3 的实测价格
  check-raster.mjs     第一道闸门：栅格化（逐像素 + 平移不变性）
  check-engine.mjs     第二道闸门：跨语言推理一致性（自动找 App 加载的模型）
  check-game.mjs       第三道闸门：玩法状态机
  make_icons.py        图标：用同一套笔刷规则画一只猫
  vectors/             三道闸门的参考数据（入库，CI 要用）
.github/
  workflows/ios.yml    在 macOS runner 上出未签名 ipa
```

画册里存的是**笔画矢量**（`[[x,y],...]`，8~20 KB）而不是渲染好的截图（1080p PNG 约 400 KB）。重绘时按当前屏幕宽度重新算路径，所以缩略图在任何设备上都清晰。

---

## 模型格式（v4）

全小端：

```
"DDMN"                    4B   magic
<B version=4, H nClasses, H nOps>
  每 op：<B type, B flags>  <H cin, H cout>  f32 wScale[]  f32 bias[cout]  int8 w[]
<H width>  f32 headScale[nClasses]  int8 headW[nClasses*width]  f32 headB[nClasses]
```

- `type`：`0` 普通 3×3，`1` depthwise 3×3，`2` pointwise 1×1
- `flags` 的 bit0：这个 op 之后接一个 2×2 最大池化
- 权重个数由 type 决定：普通 3×3 是 `cout*cin*9`，depthwise 是 `cin*9`（torch 对 `groups=cin` 只存一份权重，所以 `w.shape` 是 `(c,1,3,3)`），1×1 是 `cout*cin`
- scale 的粒度也跟着 type：普通卷积与 1×1 按输出通道一条 scale，depthwise 按输入通道一条

权重行主序。端上解量化成 `Float32Array` 常驻。

**为什么要发明一个带 op 类型的格式**：v2/v3 假设拓扑固定（4 段同构卷积），端上是 4 个硬编码的 `conv3x3` 调用。换成可分离卷积后同一个「层」里有两个算子，拓扑不再是「4 段」而是「7 个 op」，把结构写进数据里，端上就变成一个小解释器——以后再加算子（比如 5×5 或残差）不用改二进制布局。

池化点挂在「每段的最后一个 op」上，这是最容易写错的地方：可分离卷积一段有 dw+pw 两个 op，把池化挂在 dw 上等于在混通道之前就下采样，特征图会塌，而且不报错。

v3 相比 v2 改的是分类头也量化成 per-class 对称 int8（分类头是 `(148,128)` fp32、整包三成）。v4 再加 op 类型。

`parseModel` 同时认 v2 / v3 / v4，旧权重文件不会因为格式升级当场报废。

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
