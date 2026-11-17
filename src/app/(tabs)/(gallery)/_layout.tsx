/**
 * 画册所在的导航栈。大标题「画册」由 UINavigationBar 提供，
 * 滚动时自动吸顶、背景渐变成毛玻璃。
 */

import { Stack } from 'expo-router';

import { Palette, Typography } from '@/constants/design';

export default function GalleryStack() {
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
      <Stack.Screen name="gallery" options={{ title: '画册', headerLargeTitle: true }} />
    </Stack>
  );
}
