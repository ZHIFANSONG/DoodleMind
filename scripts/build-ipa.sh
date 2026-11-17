#!/usr/bin/env bash
#
# 把 Expo prebuild 生成的 ios 工程编译成**未签名** ipa。
#
# 为什么单独成脚本而不是塞在 workflow 的 run: 里：
# 这段逻辑反复出问题（scheme 取法、strip 作用对象、管道 SIGPIPE），
# 放在文件里能本地对着 shellcheck 看、能在日志里逐行定位，而不是每次
# 改完还得 push 一次等 CI 排队。
#
# 用法（在仓库根目录）：bash scripts/build-ipa.sh
# 前置：npx expo prebuild --platform ios --clean --no-install 已经跑过、Pods 已装。
set -euo pipefail

cd "$(dirname "$0")/.."

# CI 上 workspace 是相对路径，统一挪到 ios 里干活（agvtool 也要求这样）
cd ios

echo "── ios 目录 ──"
ls -la

# 取 workspace / project。用 for 而不是 `ls *.xcworkspace`。
#
# 这是一个真实的坑：`ls` 只有一个匹配项时会把它当成**目录去列内容**，于是
# `workspace=$(ls *.xcworkspace | head -1)` 拿到的不是 "DoodleMind.xcworkspace"，
# 而是它里面的第一个条目 "contents.xcworkspacedata"。接着
# `xcodebuild -workspace contents.xcworkspacedata` 就会报
# "error: 'contents.xcworkspacedata' does not exist." —— 明明文件就在那儿，
# 报错却像是文件丢了，极难看出是变量取错了。for 循环取的是匹配项本身，没有这个行为。
workspace=""
for d in *.xcworkspace; do
  if [ -d "$d" ]; then workspace="$d"; break; fi
done
project=""
for d in *.xcodeproj; do
  if [ -d "$d" ]; then project="$d"; break; fi
done
echo "workspace=[$workspace] project=[$project]"

if [ -n "$workspace" ]; then
  echo "── workspace 内容 ──"
  ls -la "$workspace" || true
fi

# 自愈：workspace 目录在、但缺 contents.xcworkspacedata，就自己写一份。
# 这不是猜——内容就是 CocoaPods 会写的那两行：client 工程 + Pods 工程。
# 之前 CI 挂在这里，xcodebuild 只丢一句 "contents.xcworkspacedata does not exist"，
# 完全看不出是"文件没生成"还是"路径给错了"；与其等它把话说明白，不如直接补齐。
if [ -n "$workspace" ] && [ -n "$project" ] && [ ! -f "$workspace/contents.xcworkspacedata" ]; then
  echo "── workspace 缺 contents.xcworkspacedata，按 CocoaPods 的格式补一份 ──"
  {
    echo '<?xml version="1.0" encoding="UTF-8"?>'
    echo '<Workspace'
    echo '   version = "1.0">'
    echo "   <FileRef"
    echo "      location = \"group:$project\">"
    echo '   </FileRef>'
    if [ -d Pods/Pods.xcodeproj ]; then
      echo '   <FileRef'
      echo '      location = "group:Pods/Pods.xcodeproj">'
      echo '   </FileRef>'
    fi
    echo '</Workspace>'
  } > "$workspace/contents.xcworkspacedata"
  cat "$workspace/contents.xcworkspacedata"
fi

# scheme 直接从磁盘上读，不经过 xcodebuild：
# `xcodebuild -list -json -workspace ...` 一旦 workspace 有问题就会先报一堆
# 无关的错误，把真正的原因埋掉。xcscheme 文件名就是 scheme 名，这是稳定的。
scheme=""
if [ -n "$project" ]; then
  for f in "$project"/xcshareddata/xcschemes/*.xcscheme; do
    if [ -f "$f" ]; then
      scheme=$(basename "$f" .xcscheme)
      break
    fi
  done
  echo "── 共享 scheme ──"
  ls -la "$project/xcshareddata/xcschemes" || true
fi
echo "scheme=[$scheme]"
test -n "$scheme"

# 关于“抹掉构建机路径”：miniWatts 用的是 OTHER_SWIFT_FLAGS / OTHER_CFLAGS 里的
# file-prefix-map。它值得保留——Swift 的 assert 消息会把 #file 的绝对路径当字面量
# 编进二进制，包内就会留下 /Users/runner/work/...。
#
# 但要注意写法：Xcode 把 build setting 的值按**空格**切成 argv，不按 `=` 切。
# 所以 `OTHER_SWIFT_FLAGS=-file-prefix-map=/a=/b` 会让 swift driver 收到一个
# 完整的 `-file-prefix-map=/a=/b` token，它只认 `-file-prefix-map` + 一个参数，
# 于是报 unknown argument。必须写成空格分隔。
# C 侧的 `-ffile-prefix-map=/a=/b` 是单 token 合法写法，不用拆。
common=(
  -scheme "$scheme"
  -configuration Release
  -destination 'generic/platform=iOS'
  -derivedDataPath build
  CODE_SIGNING_ALLOWED=NO
  CODE_SIGNING_REQUIRED=NO
  CODE_SIGN_IDENTITY=""
  CODE_SIGN_ENTITLEMENTS=""
  DEVELOPMENT_TEAM=""
  DEBUG_INFORMATION_FORMAT=dwarf
  SWIFT_COMPILATION_MODE=wholemodule
)
map_flags=(
  "OTHER_SWIFT_FLAGS=-file-prefix-map $PWD=/DoodleMind"
  "OTHER_CFLAGS=-ffile-prefix-map=$PWD=/DoodleMind"
)

run_build() {
  xcodebuild build "$@" > xcodebuild.log 2>&1 || true
  grep -q 'BUILD SUCCEEDED' xcodebuild.log
}

if [ -n "$workspace" ] && [ -f "$workspace/contents.xcworkspacedata" ]; then
  target_flags=(-workspace "$workspace")
else
  target_flags=(-project "$project")
fi

# 逐级降级重试。每一条都对应一个真出现过的失败，而不是预防性猜测：
# 一轮 macOS runner 排队 + 编译要十几分钟，为一个能在本地判断的兼容性问题
# 白跑一轮是最亏的。
succeeded=0
if run_build "${target_flags[@]}" "${common[@]}" "${map_flags[@]}"; then
  succeeded=1
fi

if [ "$succeeded" = 0 ] && grep -q 'unknown argument.*file-prefix-map' xcodebuild.log; then
  echo "── 该 toolchain 不认 file-prefix-map，去掉重试 ──"
  if run_build "${target_flags[@]}" "${common[@]}"; then succeeded=1; fi
fi

# expo-modules-jsi 在 Swift 6 语言模式下会被判 "sending 'slot' risks causing
# data races"（区域隔离，SE-0414）。那是**依赖**里的代码，不该由我们改；
# 而且我们的 App 是单线程时序逻辑，没有真并发。降回 Swift 5 语言模式 + minimal。
if [ "$succeeded" = 0 ] && grep -qE 'risks causing data races|requires that .* be Sendable|Swift 6 language mode' xcodebuild.log; then
  echo "── 依赖在 Swift 6 严格并发下过不了，降回 Swift 5 语言模式重试 ──"
  if run_build "${target_flags[@]}" "${common[@]}" "${map_flags[@]}" \
      SWIFT_VERSION=5.0 SWIFT_STRICT_CONCURRENCY=minimal SWIFT_UPCOMING_FEATURE_MEMBER_IMPORT_VISIBILITY=NO; then
    succeeded=1
  fi
fi

grep -E 'error:|BUILD ' xcodebuild.log | tail -40 || true
grep -q 'BUILD SUCCEEDED' xcodebuild.log || {
  echo "编译失败，末尾日志："
  tail -80 xcodebuild.log
  exit 1
}

app=$(find build/Build/Products/Release-iphoneos -maxdepth 1 -name '*.app' | head -1)
echo "产物 $app"
test -d "$app"

# 去掉签名残留与调试符号里的绝对路径。strip 要作用在**可执行文件**上：
# 直接对 .app 目录调会被当成打不开的文件、静默失败（那行还带了 || true，
# 等于什么都没剥）。
rm -rf "$app/_CodeSignature"
exe=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$app/Info.plist")
xcrun strip -S -x "$app/$exe"
find "$app" -name '*.dSYM' -prune -exec rm -rf {} + 2>/dev/null || true

rm -rf Payload DoodleMind-unsigned.ipa
mkdir Payload
cp -R "$app" Payload/
zip -qry DoodleMind-unsigned.ipa Payload
ls -lh DoodleMind-unsigned.ipa
