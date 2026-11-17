import { Tabs } from 'expo-router';
import { SymbolView } from 'expo-symbols';

import { Palette, Typography } from '@/constants/design';

/**
 * 为什么不用 NativeTabs：
 * NativeTabs 走系统 UITabBar，图标必须是 SF Symbol 或图片资源，样式几乎不可改。
 * DoodleMind 要的是「纸与墨」的统一观感，所以用 JS tabs 自己画，
 * 顺便能在中间塞一颗朱红色的「新建」按钮。
 */
export default function AppTabs() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: Palette.seal,
        tabBarInactiveTintColor: Palette.inkFaint,
        tabBarStyle: {
          backgroundColor: Palette.paper,
          borderTopColor: Palette.rule,
          borderTopWidth: 1,
        },
        tabBarLabelStyle: {
          fontFamily: Typography.zh,
          fontSize: 11,
          fontWeight: '600',
        },
        sceneStyle: { backgroundColor: Palette.paper },
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: '开始',
          tabBarIcon: ({ color, size }) => (
            <SymbolView
              name="pencil.and.outline"
              tintColor={color}
              size={size}
              resizeMode="scaleAspectFit"
            />
          ),
        }}
      />
      <Tabs.Screen
        name="gallery"
        options={{
          title: '画册',
          tabBarIcon: ({ color, size }) => (
            <SymbolView name="square.grid.2x2" tintColor={color} size={size} resizeMode="scaleAspectFit" />
          ),
        }}
      />
      <Tabs.Screen
        name="about"
        options={{
          title: '关于',
          tabBarIcon: ({ color, size }) => (
            <SymbolView name="cpu" tintColor={color} size={size} resizeMode="scaleAspectFit" />
          ),
        }}
      />
    </Tabs>
  );
}
