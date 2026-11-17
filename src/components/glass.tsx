/**
 * 毛玻璃 / 液态玻璃统一层。
 *
 * 候选栏是这局游戏的信息核心，它的底必须真的「透」——玩家要能透过它看到自己
 * 刚画的那一笔，才感觉得到 AI 正贴在画布上看。
 *
 * 三个平台三条路：
 *  - iOS 26+：系统的 Liquid Glass（`expo-glass-effect`），会跟着系统音量键、
 *    控制中心那套材质一起变，且能折射背后的墨色。这是苹果自己画的东西，JS 画不出来。
 *  - iOS 18 及以下：`expo-blur` 的 UIBlurEffect。玻璃没有折射，但模糊是真的。
 *  - web / Android：`expo-blur` 退化成一个半透明色块（它的非 iOS 实现就是
 *    一个带 `backgroundColor` 的 View），此时必须叠一层纸色蒙版，
 *    否则白底上的白字会直接消失。
 *
 * 为什么不在 iOS 上一律用 GlassView：`isGlassEffectAPIAvailable()` 在部分
 * iOS 26 beta 上返回 false，直接渲染 GlassView 会崩。必须逐次检查。
 */

import { BlurView } from 'expo-blur';
import { GlassView, isGlassEffectAPIAvailable } from 'expo-glass-effect';
import { Platform, StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';

import { Palette } from '@/constants/design';

/** 只在 iOS 上问一次原生模块；非 iOS 平台这个模块根本不存在，不能 import 到调用点。 */
const GLASS_OK = Platform.OS === 'ios' ? safeCheck() : false;

function safeCheck(): boolean {
  try {
    return isGlassEffectAPIAvailable();
  } catch {
    // 原生模块没链进来（比如 Expo Go 里跑、或 pod 没装）时 requireNativeModule 会抛
    return false;
  }
}

type Props = {
  style?: StyleProp<ViewStyle>;
  /** 玻璃强度，只在 BlurView 分支生效；Liquid Glass 的浓度由系统决定 */
  intensity?: number;
  children?: React.ReactNode;
};

export function Glass({ style, intensity = 42, children }: Props) {
  if (GLASS_OK) {
    return (
      <GlassView glassEffectStyle="regular" colorScheme="light" style={style}>
        {children}
      </GlassView>
    );
  }

  if (Platform.OS === 'ios') {
    return (
      <BlurView intensity={intensity} tint="systemUltraThinMaterialLight" style={style}>
        {children}
      </BlurView>
    );
  }

  // web / Android：BlurView 在这两个平台不模糊，只是一层带背景色的 View。
  // 补一层纸色蒙版把背后的内容压淡，观感才接近「毛玻璃」而不是「透明块」。
  return (
    <View style={[style, styles.fallback]}>
      <BlurView intensity={intensity} tint="light" style={StyleSheet.absoluteFill} />
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  fallback: {
    backgroundColor: 'rgba(251,250,246,0.72)',
    overflow: 'hidden',
    // 蒙版色就是纸色，加一条极淡的分隔线让它在白底上仍有边界
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: Palette.rule,
  },
});
