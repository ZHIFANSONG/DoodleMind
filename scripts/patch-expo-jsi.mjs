#!/usr/bin/env node
/**
 * 修 expo-modules-jsi 在 Xcode 26 的 Swift 6 语言模式下编不过的那一行。
 *
 * ## 问题
 *
 *   apple/Sources/ExpoModulesJSI/Runtime/Values/JavaScriptObject.swift
 *     error: sending 'slot' risks causing data races      (SE-0414 区域隔离)
 *
 * 代码是这样的：
 *
 *   nonisolated(unsafe) let slot = buffer.baseAddress!          // UnsafeMutablePointer，非 Sendable
 *   try runtime._withUnsafeGuaranteedRef { runtime in
 *     try JavaScriptActor.assumeIsolated {
 *       try encode(JavaScriptValue.Slot(slot), runtime)          // ← 这一行被诊断
 *     }
 *   }
 *
 * 它是有意在同一线程上把裸指针交给 actor 隔离的闭包（源码注释写明了
 * "The isolated closure below runs synchronously on this thread"），
 * 所以静态判定在这里是误报；但它是 error，编不过就没有 ipa。
 *
 * ## 为什么不能在 xcodebuild 命令行上想办法
 *
 * 这个文件不是我们的 target 编的。ExpoModulesJSI.podspec 有个 script phase 跑
 * apple/scripts/build-xcframework.sh，脚本里
 *
 *   (cd "$PACKAGE_DIR" && env -i PATH=... xcodebuild build -scheme ExpoModulesJSI ...)
 *
 * 清空环境另起一个 xcodebuild 编 SPM 包。外层传 SWIFT_VERSION / SWIFT_STRICT_CONCURRENCY
 * 一律进不去（env -i 就是清环境）；换 Xcode 26.0.1 也一样报。唯一有效的是改磁盘上的源码。
 *
 * ## 为什么不是把 Package.swift 的 swiftLanguageModes 降到 .v5
 *
 * 试过：那条路会引出另外两个错误——
 *   JavaScriptRuntime.swift:439 '$' is not a valid digit in integer literal
 *   JavaScriptPromise.swift:59  call to global actor 'JavaScriptActor'-isolated initializer
 *                               'init()' in a synchronous nonisolated context
 * 也就是说这个包是按 Swift 6 语义写的，降模式并不是"少检查一点"，而是换了一套规则。
 *
 * ## 修法
 *
 * 只把跨隔离边界的那个指针装进一个 @unchecked Sendable 的盒子。运行时行为完全不变
 * （还是同一个指针、同一次同步调用），只是让区域隔离检查看到的是一个 Sendable 捕获。
 *
 * 幂等；任何一个替换没匹配上都直接报错，免得"以为打了补丁"又白等一轮 CI。
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const file = join(
  root,
  'node_modules',
  'expo-modules-jsi',
  'apple',
  'Sources',
  'ExpoModulesJSI',
  'Runtime',
  'Values',
  'JavaScriptObject.swift',
);

const MARK = 'DoodleMind-SE0414-workaround';

if (!existsSync(file)) {
  console.error(`[patch-jsi] 找不到 ${file}`);
  process.exit(1);
}

let src = readFileSync(file, 'utf8');

if (src.includes(MARK)) {
  console.log('[patch-jsi] 已打过补丁，跳过');
  process.exit(0);
}

// 1) 在文件末尾追加盒子类型。
const box = `

// MARK: - ${MARK}
//
// 把 UnsafeMutablePointer 越过一次隔离边界。\`JavaScriptObject.withEncodedValue\` 里的
// 那个指针本来就在同一线程上同步交给 @JavaScriptActor 闭包（源码注释写明了），但
// SE-0414 的区域隔离只看类型：非 Sendable 的值进 actor 闭包就是 "sending ... risks
// causing data races"。用 @unchecked Sendable 显式声明"这里由我负责"，运行时行为不变。
private struct DoodleMindUnsafeSlotBox: @unchecked Sendable {
  let pointer: UnsafeMutablePointer<facebook.jsi.Value>
  init(_ pointer: UnsafeMutablePointer<facebook.jsi.Value>) {
    self.pointer = pointer
  }
}
`;
src = src.trimEnd() + box;

// 2) 在调用前把指针装盒。
const beforeWrap = `    try runtime._withUnsafeGuaranteedRef { runtime in
      try JavaScriptActor.assumeIsolated {
        try encode(JavaScriptValue.Slot(slot), runtime)
      }
    }`;
const afterWrap = `    // ${MARK}: 见文件末尾的 DoodleMindUnsafeSlotBox
    let unsafeSlot = DoodleMindUnsafeSlotBox(slot)
    try runtime._withUnsafeGuaranteedRef { runtime in
      try JavaScriptActor.assumeIsolated {
        try encode(JavaScriptValue.Slot(unsafeSlot.pointer), runtime)
      }
    }`;

if (!src.includes(beforeWrap)) {
  console.error('[patch-jsi] 没匹配到 withEncodedValue 里的 assumeIsolated 调用段，');
  console.error('[patch-jsi] expo-modules-jsi 可能改过这段代码，请重新确认编译失败原因。');
  process.exit(1);
}
src = src.replace(beforeWrap, afterWrap);

writeFileSync(file, src);
console.log('[patch-jsi] 已打补丁：withEncodedValue 的指针改为经 @unchecked Sendable 盒子跨隔离边界');
