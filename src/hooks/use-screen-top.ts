/**
 * 页面顶部要额外让出的距离。
 *
 * iOS / Android：0。因为这三个 tab 页现在各自套在 `Stack` 里，顶部的
 * UINavigationBar 已经把状态栏区和标题吃掉了——再补 `insets.top` 就会空出一条
 * 和导航栏等高的白带（这个 bug 在改大标题时很容易被引入，因为页面本身看起来没错，
 * 只是内容整体下移了一屏的 1/12）。
 *
 * web：web 端 `NativeTabs` 渲染的是一个 `position: fixed; top: 24px` 的悬浮胶囊
 * （见 `node_modules/expo-router/assets/native-tabs.module.css` 的 `.navigationMenuRoot`），
 * **不占文档流**，所以内容必须自己往下让 84pt 才不会被压住。
 */

import { Platform } from 'react-native';

/** 悬浮胶囊顶 24 + 自身高 40 + 视觉间隙 20。 */
const WEB_TAB_BAR_TOP = 84;

export function useScreenTop(): number {
  return Platform.OS === 'web' ? WEB_TAB_BAR_TOP : 0;
}
