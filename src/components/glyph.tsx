/**
 * 图标层。
 *
 * 为什么不用 `expo-symbols` 直接铺满全项目：`SymbolView` 的 name 只有在 iOS 上
 * 才接受 SF Symbol 字符串。看它自己的非 iOS 实现（`expo-symbols/build/SymbolView.js`）：
 *
 *   const name = typeof nameProp === 'object' ? nameProp[Platform.OS === 'android' ? 'android' : 'web'] : null;
 *   if (!name) return <>{fallback}</>;
 *
 * 传字符串在 web / Android 上直接 `null`，而我们没传 `fallback` → 渲染成一块空的灰方块。
 * 这个 App 要在 web 上预览界面（Windows 开发没有 Xcode），空图标会让人误判版式。
 *
 * 所以这里做一层：iOS 继续用真正的 SF Symbols（原生渲染，灰度、光学尺寸都是系统级的），
 * 其它平台用 react-native-svg 现画同一套线性图形。两边视觉语言一致（1.8 描边、圆头、24 网格），
 * 但都不再依赖「字体加载成功」这种运行时条件。
 */

import { SymbolView } from 'expo-symbols';
import { Platform, type ColorValue } from 'react-native';
import Svg, { Circle, Path, Rect } from 'react-native-svg';

export type GlyphName =
  | 'pencil'
  | 'grid'
  | 'chip'
  | 'chevronRight'
  | 'chevronLeft'
  | 'eyeSlash'
  | 'scribble'
  | 'emptyGallery'
  | 'dot';

/** iOS 上对应的 SF Symbol。名字都要能被 sf-symbols-typescript 校验。 */
const SF_NAME: Record<GlyphName, string> = {
  pencil: 'pencil.and.outline',
  grid: 'square.grid.2x2',
  chip: 'cpu',
  chevronRight: 'chevron.right',
  chevronLeft: 'chevron.left',
  eyeSlash: 'eye.slash',
  scribble: 'scribble',
  emptyGallery: 'rectangle.and.pencil.and.ellipsis',
  dot: 'circle.fill',
};

type Props = {
  name: GlyphName;
  size?: number;
  color: ColorValue;
  /**
   * 让图标动一下。只有 iOS 上生效（SF Symbols 的原生动画，由系统按符号的分层结构
   * 决定每一层怎么动——`bounce` 会让铅笔「点一下」，`pulse` 是呼吸，`scale` 是放大再回弹）。
   * 判中/判输用它做结果反馈：比整块卡片缩放更细，也更像系统自己的反馈。
   */
  effect?: 'bounce' | 'pulse' | 'scale';
  /**
   * SF Symbols 的视觉变体。`hierarchical` 会用同一色相分出层次（比如实心+描边），
   * 单色符号上看起来更有体积感。
   */
  variant?: 'monochrome' | 'hierarchical' | 'palette' | 'multicolor';
};

export function Glyph({ name, size = 20, color, effect, variant }: Props) {
  if (Platform.OS === 'ios') {
    return (
      <SymbolView
        name={SF_NAME[name] as never}
        tintColor={color}
        size={size}
        resizeMode="scaleAspectFit"
        type={variant ?? 'monochrome'}
        // animationSpec 的 key 必须只在值变化时才变，否则每次渲染都会重播动画。
        // 这里让 effect 本身决定：换成别的 effect 就重播一次，同一个 effect 不重播。
        animationSpec={effect ? { effect: { type: effect }, repeating: false } : undefined}
      />
    );
  }
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      {draw(name, typeof color === 'string' ? color : '#17161A')}
    </Svg>
  );
}

function draw(name: GlyphName, c: string) {
  const stroke = { stroke: c, strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  switch (name) {
    // 一支斜插的铅笔：笔尖在左下，笔尾在右上，中间一条削笔线
    case 'pencil':
      return (
        <>
          <Path d="M4.6 19.4h3.2L19.4 7.8a2.1 2.1 0 0 0 0-3l-0.6-0.6a2.1 2.1 0 0 0-3 0L4.6 16.2z" {...stroke} />
          <Path d="M14.6 5.4l4 4" {...stroke} />
        </>
      );
    // 四宫格：四个小方角，中间留缝——比实心方块更像「画册」而不是「菜单」
    case 'grid':
      return (
        <>
          <Rect x="4" y="4" width="6.6" height="6.6" rx="1.6" {...stroke} />
          <Rect x="13.4" y="4" width="6.6" height="6.6" rx="1.6" {...stroke} />
          <Rect x="4" y="13.4" width="6.6" height="6.6" rx="1.6" {...stroke} />
          <Rect x="13.4" y="13.4" width="6.6" height="6.6" rx="1.6" {...stroke} />
        </>
      );
    // 芯片：外框 + 内芯 + 四条引脚，用来标「模型/推理」这一页
    case 'chip':
      return (
        <>
          <Rect x="6.4" y="6.4" width="11.2" height="11.2" rx="2.2" {...stroke} />
          <Rect x="10" y="10" width="4" height="4" rx="1" {...stroke} />
          <Path d="M9.6 3.4v3M14.4 3.4v3M9.6 17.6v3M14.4 17.6v3M3.4 9.6h3M3.4 14.4h3M17.6 9.6h3M17.6 14.4h3" {...stroke} />
        </>
      );
    case 'chevronRight':
      return <Path d="M9 5l7 7-7 7" {...stroke} />;
    case 'chevronLeft':
      return <Path d="M15 5l-7 7 7 7" {...stroke} />;
    // 被划掉的眼睛：上半弧 + 瞳孔 + 一道斜杠
    case 'eyeSlash':
      return (
        <>
          <Path d="M3.6 12s3.2-5.2 8.4-5.2 8.4 5.2 8.4 5.2-3.2 5.2-8.4 5.2S3.6 12 3.6 12z" {...stroke} />
          <Circle cx="12" cy="12" r="2.4" {...stroke} />
          <Path d="M4.4 20l15.2-16" {...stroke} />
        </>
      );
    // 涂鸦：两段不规则波线，代表「随便画」
    case 'scribble':
      return (
        <>
          <Path d="M3.8 15.2c1.6-4.4 3.4-4.6 4.4-1.4s2.2 3.6 3.4-0.2 2.6-4.4 3.6-0.6 2 3.4 3.2 0.6" {...stroke} />
          <Path d="M6.4 19.4c1.2-2.6 2.6-2.8 3.4-0.8" {...stroke} />
        </>
      );
    // 空画册：一个方框 + 一支伸出来的铅笔
    case 'emptyGallery':
      return (
        <>
          <Rect x="3.4" y="6.4" width="12.4" height="14.2" rx="2.4" {...stroke} />
          <Path d="M7.2 16.2l2.6-3.4 2 2.2 1.6-1.8 2.4 3" {...stroke} />
          <Path d="M17.2 4.6l2.4 2.4-5 5-2.8 0.4 0.4-2.8z" {...stroke} />
        </>
      );
    case 'dot':
      return <Circle cx="12" cy="12" r="5" fill={c} />;
  }
}
