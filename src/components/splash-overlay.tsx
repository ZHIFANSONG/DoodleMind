/**
 * 启动过渡。
 *
 * 模板自带的是 Expo logo 缩放动画——必须换掉，否则用户第一眼看到的是别人的品牌。
 * 这里只做一件极简的事：纸面上一个墨点在 320ms 内洇开，「DoodleMind」随后淡入，
 * 然后整层淡出。用 RN 内置 Animated 而不是 reanimated：这段动画只跑一次、
 * 不涉及手势和共享值，用 reanimated 反而多一层 worklet 调度。
 */

import { useEffect, useState } from 'react';
import { Animated, Easing, Platform, StyleSheet } from 'react-native';
import * as SplashScreen from 'expo-splash-screen';

import { Palette, Typography } from '@/constants/design';

// web 上没有系统闪屏，这一层纯属多余：它只会让每次刷新都白等一秒，
// 而 web 端唯一的用途是本地预览 UI，任何多余等待都是干扰。直接不进这个分支。
const ENABLED = Platform.OS !== 'web';

export function SplashOverlay() {
  const [visible, setVisible] = useState(ENABLED);
  // 用 useState 的惰性初始化持有 Animated.Value：它一旦创建就不该被替换，
  // 用 useRef 存会在渲染里读 ref.current，React 的 lint 会（正确地）报错。
  const [ink] = useState(() => new Animated.Value(0));
  const [word] = useState(() => new Animated.Value(0));

  useEffect(() => {
    // 先让系统闪屏退场，再播我们自己的过渡，避免两层叠在一起
    SplashScreen.hideAsync().catch(() => {});
    if (!ENABLED) return;
    Animated.sequence([
      Animated.timing(ink, {
        toValue: 1,
        duration: 320,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }),
      Animated.timing(word, {
        toValue: 1,
        duration: 260,
        easing: Easing.out(Easing.quad),
        useNativeDriver: true,
      }),
      Animated.delay(200),
      Animated.timing(ink, {
        toValue: 0,
        duration: 260,
        easing: Easing.in(Easing.quad),
        useNativeDriver: true,
      }),
    ]).start(() => setVisible(false));
  }, [ink, word]);

  if (!visible) return null;

  return (
    <Animated.View
      style={[styles.overlay, { opacity: ink.interpolate({ inputRange: [0, 0.15, 1], outputRange: [0, 1, 1] }) }]}
      pointerEvents="none">
      <Animated.View
        style={[
          styles.blot,
          { transform: [{ scale: ink.interpolate({ inputRange: [0, 1], outputRange: [0.4, 1] }) }] },
        ]}
      />
      <Animated.Text style={[styles.word, { opacity: word }]}>DoodleMind</Animated.Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: Palette.paper,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  blot: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: Palette.ink,
  },
  word: {
    position: 'absolute',
    marginTop: 96,
    color: Palette.ink,
    fontSize: 17,
    fontFamily: Typography.title,
    letterSpacing: 0.4,
  },
});
