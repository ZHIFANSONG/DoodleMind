import { DefaultTheme, Stack, ThemeProvider } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';

import { SplashOverlay } from '@/components/splash-overlay';
import { Palette, Typography } from '@/constants/design';

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

/**
 * 根导航壳只有两层：三个常驻 tab（`(tabs)`）和一局游戏（`play`）。
 *
 * 为什么要把它们分层：游戏页要占满整屏、盖住 tab 栏，且它是个有时限的
 * 「进行中的一局」，退出即作废——这是 Stack 推入的语义，不是 tab 切换的语义。
 * 早先把 `play` 直接挂在 Tabs 下、用 `href: null` 摘出序列，看着能跑，
 * 但 web 上 URL 直达会静默退回首页（tab 容器里没有可激活的子路由）。
 */
export default function RootLayout() {
  return (
    <ThemeProvider value={paperTheme}>
      <SplashOverlay />
      <Stack
        screenOptions={{
          contentStyle: { backgroundColor: Palette.paper },
          headerStyle: { backgroundColor: Palette.paper },
          headerShadowVisible: false,
          headerTintColor: Palette.seal,
          headerTitleStyle: {
            color: Palette.ink,
            fontFamily: Typography.zh,
            fontSize: 16,
            fontWeight: '600',
          },
          headerBackTitle: '回去',
        }}>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        {/*
          `gestureEnabled: false` 是必须的，不是偏好。

          这一屏整块都是画布：手指从左边缘往里划是在画线，系统却把它当成
          「返回上一页」的交互式转场——两条手势抢同一次触摸，结果是画到一半
          被推出这一局，而且越靠左画越容易被抢走。iOS 自己的画图类 App 也都
          关掉这个手势，退出走导航栏左上角的返回按钮（系统那个 chevron + 回去）。
        */}
        <Stack.Screen
          name="play"
          options={{ animation: 'slide_from_bottom', gestureEnabled: false }}
        />
      </Stack>
    </ThemeProvider>
  );
}
