import { DefaultTheme, ThemeProvider } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';

import AppTabs from '@/components/app-tabs';
import { SplashOverlay } from '@/components/splash-overlay';
import { Palette } from '@/constants/design';

SplashScreen.preventAutoHideAsync();

/**
 * 只提供浅色主题。
 *
 * 理由是「纸与墨」这套观感建立在暖白纸面上：切到深色模式时，
 * 墨线和朱红印章在近黑背景上会变成发光色块，整个设计语言就崩了。
 * 与其做一个糟糕的深色版，不如锁死一个做好的浅色版——
 * 这也是很多以纸张为隐喻的工具类 App（笔记、素描）的通行做法。
 */
const paperTheme = {
  ...DefaultTheme,
  colors: {
    ...DefaultTheme.colors,
    background: Palette.paper,
    card: Palette.paper,
    text: Palette.ink,
    border: Palette.rule,
    primary: Palette.seal,
  },
};

export default function RootLayout() {
  return (
    <ThemeProvider value={paperTheme}>
      <SplashOverlay />
      <AppTabs />
    </ThemeProvider>
  );
}
