/**
 * 首页所在的导航栈。
 *
 * 为什么每个 tab 各自套一层 `Stack`：iOS 的大标题（large title）不是一行文字，
 * 它是 UINavigationBar 的一个状态——滚动时标题缩小并吸顶、背景从透明渐变成毛玻璃、
 * 状态栏文字跟着变深变浅。这些行为只有把内容真的放进 `UINavigationController`
 * 里才会发生。所以首页不再自己画「关于」两个大字，而是 `title` + `largeTitle`。
 *
 * 这一层也顺便给了每个 tab 自己的导航栈：以后在首页里 push 一个详情页，
 * 返回时会回到首页而不是跳到别的 tab。
 */

import { Stack } from 'expo-router';

import { Palette, Typography } from '@/constants/design';

export default function HomeStack() {
  return (
    <Stack
      screenOptions={{
        headerStyle: { backgroundColor: Palette.paper },
        headerShadowVisible: false,
        headerTintColor: Palette.seal,
        headerTitleStyle: {
          color: Palette.ink,
          fontFamily: Typography.zh,
          fontSize: 17,
          fontWeight: '600',
        },
        headerLargeTitleStyle: {
          color: Palette.ink,
          fontFamily: Typography.zh,
          fontSize: 30,
          fontWeight: '700',
        },
        contentStyle: { backgroundColor: Palette.paper },
      }}>
      <Stack.Screen name="index" options={{ title: 'DoodleMind', headerLargeTitle: true }} />
    </Stack>
  );
}
