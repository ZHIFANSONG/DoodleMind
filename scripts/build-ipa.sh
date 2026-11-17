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

# 构建参数。Pods 已经装好，正常情况必须走 workspace——只给 -project 会
# 找不到 Pods 的 target，链接阶段报一堆 undefined symbol。
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
  # 把构建机的家目录从调试信息里抹掉，否则包里会留下 /Users/runner/work/...
  "OTHER_SWIFT_FLAGS=-file-prefix-map=$PWD=/DoodleMind"
  "OTHER_CFLAGS=-ffile-prefix-map=$PWD=/DoodleMind"
)

if [ -n "$workspace" ] && [ -f "$workspace/contents.xcworkspacedata" ]; then
  echo "── 用 workspace 构建 ──"
  xcodebuild build -workspace "$workspace" "${common[@]}" > xcodebuild.log 2>&1 || true
elif [ -n "$project" ]; then
  # 兜底：workspace 缺失或不完整时退回 project。Pods 情况下链接大概率失败，
  # 但至少能看到真正的编译错误，而不是卡在取 scheme 那一步。
  echo "── workspace 不可用，退回 project 构建 ──"
  xcodebuild build -project "$project" "${common[@]}" > xcodebuild.log 2>&1 || true
else
  echo "既没有 .xcworkspace 也没有 .xcodeproj" >&2
  exit 1
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
