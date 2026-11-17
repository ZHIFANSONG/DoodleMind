import { NativeTabs } from 'expo-router/native-tabs';
import type { SFSymbol } from 'sf-symbols-typescript';

import { Palette } from '@/constants/design';

/**
 * 三个常驻页的 tab 栏 —— 用**真正的系统 UITabBar**，不是 JS 画的。
 *
 * 为什么从 JS `Tabs` 换成 `NativeTabs`：
 * 1. 手感。系统 tab bar 的点击高亮、长按、VoiceOver、动态字体缩放、
 *    「点当前 tab 回到顶部」这些行为，JS 版一律没有，而用户对它们的期待是免费的。
 * 2. 观感。iOS 26 起 tab bar 的玻璃质感由系统绘制（`backgroundColor` 直接失效），
 *    自己画的实色栏会在这套系统上显得又平又假。
 * 3. 布局。系统栏会自动让出安全区、跟随键盘、支持 sidebarAdaptable（iPad 转侧栏）。
 *
 * 代价是样式几乎不可改：图标只能是 SF Symbol / 图片 / xcasset，颜色只能给 tint。
 * 「纸与墨」在这里靠三个参数维持——选中色用朱红、未选中用淡墨、字体用系统中文。
 */
export default function TabsLayout() {
  return (
    <NativeTabs
      tintColor={Palette.seal}
      iconColor={{ default: Palette.inkFaint, selected: Palette.seal }}
      labelStyle={{
        default: { color: Palette.inkFaint, fontSize: 10.5, fontWeight: '600' },
        selected: { color: Palette.seal, fontSize: 10.5, fontWeight: '700' },
      }}
      // iPad / 横屏大屏上允许系统把 tab 栏转成侧栏，这是系统给的免费增益。
      sidebarAdaptable
      // 滚到底部时不强制透明：画册是长列表，栏体保持玻璃才有层次。
      disableTransparentOnScrollEdge
      // 回到首页不做 popToRoot 之外的花样，但重复点当前 tab 回顶部是用户习惯。
      minimizeBehavior="onScrollDown">
      <NativeTabs.Trigger name="(home)">
        <NativeTabs.Trigger.Icon sf={{ default: 'pencil.and.outline', selected: 'pencil.and.outline' }} />
        <NativeTabs.Trigger.Label>开始</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>

      <NativeTabs.Trigger name="(gallery)">
        <NativeTabs.Trigger.Icon sf={GALLERY_FILLED} />
        <NativeTabs.Trigger.Label>画册</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>

      <NativeTabs.Trigger name="(about)">
        <NativeTabs.Trigger.Icon sf={{ default: 'cpu', selected: 'cpu.fill' }} />
        <NativeTabs.Trigger.Label>关于</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>
    </NativeTabs>
  );
}

/**
 * 画册的图标。`square.grid.2x2` 没有 filled 变体（SF Symbols 里只有 2x2 无填充版，
 * 带填充的是 `square.grid.2x2.fill`），这里分开写是为了让选中/未选中都有意义：
 * 未选中用线框、选中用实心格，比单纯改颜色更容易一眼看出当前在哪一页。
 */
const GALLERY_FILLED = {
  default: 'square.grid.2x2',
  selected: 'square.grid.2x2.fill',
} as { default: SFSymbol; selected: SFSymbol };
