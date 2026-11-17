#!/usr/bin/env node
/**
 * 把 expo-modules-jsi 的 SPM 包从 Swift 6 语言模式降到 5。
 *
 * 为什么需要这个：`expo-modules-jsi@58.0.23` 的
 *   apple/Sources/ExpoModulesJSI/Runtime/Values/JavaScriptObject.swift:946
 * 在 Xcode 26 的 Swift 6 语言模式下过不了编译：
 *   error: sending 'slot' risks causing data races
 * 那是 SE-0414 区域隔离的判定。代码本身是有意在同一线程上把裸指针交给
 * 一个 @JavaScriptActor 闭包，注释里写明了这一点，所以静态检查是误报；
 * 但它是 error，编不过就出不了 ipa。
 *
 * 为什么不能在 xcodebuild 命令上加 SWIFT_VERSION=5.0 了事：
 * 这个文件不是我们的 target 编的。ExpoModulesJSI.podspec 里有个 script phase
 * 跑 apple/scripts/build-xcframework.sh，那个脚本用
 *
 *   (cd "$PACKAGE_DIR" && env -i PATH=... xcodebuild build -scheme ExpoModulesJSI ...)
 *
 * 另起一个 xcodebuild 编译 SPM 包 —— `env -i` 就是清空环境，外层 build setting
 * 一律进不去。唯一能影响到它的是磁盘上的 Package.swift 本身。
 *
 * 降到 Swift 5 语言模式只放宽静态并发检查，不改任何运行时语义：
 * 生成的代码、调用约定、线程模型都一样。
 *
 * 幂等：已经改过就什么都不做。找不到预期的声明会明确报错，避免"以为打了补丁
 * 其实没有"，白等一轮 CI。
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const file = join(root, 'node_modules', 'expo-modules-jsi', 'apple', 'Package.swift');

if (!existsSync(file)) {
  console.error(`[patch-jsi] 找不到 ${file}`);
  process.exit(1);
}

const before = readFileSync(file, 'utf8');

// 两种写法都见过：新版本是 swiftLanguageModes，老版本是 swiftLanguageVersions。
const patterns = [
  [/(swiftLanguageModes:\s*\[.*?)\.v6/, '$1.v5'],
  [/(swiftLanguageVersions:\s*\[.*?)\.v6/, '$1.v5'],
];

let after = before;
let patched = false;
for (const [re, rep] of patterns) {
  if (re.test(after)) {
    after = after.replace(re, rep);
    patched = true;
  }
}

if (after === before) {
  if (/swiftLanguageModes:\s*\[\s*\.v5\s*\]|swiftLanguageVersions:\s*\[\s*\.v5\s*\]/.test(before)) {
    console.log('[patch-jsi] 已经是 Swift 5 语言模式，跳过');
    process.exit(0);
  }
  console.error('[patch-jsi] Package.swift 里没找到 swiftLanguageModes/swiftLanguageVersions 的 .v6 声明');
  console.error('[patch-jsi] expo-modules-jsi 可能换结构了，请重新确认编译失败原因');
  process.exit(1);
}

writeFileSync(file, after);
const changed = after
  .split('\n')
  .filter((l, i) => l !== before.split('\n')[i])
  .map((l) => l.trim());
console.log(`[patch-jsi] 已打补丁：${changed.join(' | ')}`);
if (!patched) process.exit(1);
