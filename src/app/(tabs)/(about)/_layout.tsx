/**
 * 关于所在的导航栈。大标题「关于」由系统绘制。
 */

import { Stack } from 'expo-router';

import { Palette, Typography } from '@/constants/design';

export default function AboutStack() {
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
      <Stack.Screen name="about" options={{ title: '关于', headerLargeTitle: true }} />
    </Stack>
  );
}
