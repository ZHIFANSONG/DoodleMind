/**
 * 游戏页面。三种玩法共用一具骨架，但**结算语义完全不同**，所以状态机在
 * `lib/session.ts` 里按模式分派，这里只负责渲染与交互。
 *
 * 版式从上到下按「信息密度递减」排：
 *   1. 顶栏   —— 剩余秒数 + 模式名，扫一眼就知道局势
 *   2. 题目卡 —— 目标词；盲区模式这里还要显示「别被认出来」的红字警示
 *   3. 画布   —— 正方形，吃掉剩余全部空间。画布越大模型越准，这是硬约束
 *   4. 候选栏 —— 毛玻璃，AI 的实时注意力分布，是整个游戏的信息核心
 *   5. 工具栏 —— 颜色 / 撤销 / 清空 / 放弃
 *
 * 为什么画布必须正方形：模型输入是 32x32，QuickDraw 数据也全是正方形构图。
 * 非正方形画布会在栅格化时按包围盒等比缩放，内容被压扁，模型判断悄悄变差，
 * 而且极难察觉——这种 bug 不会有任何报错，只会让人觉得「这 AI 有点笨」。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Animated,
  Easing,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
  type LayoutChangeEvent,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Stack, useLocalSearchParams, useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { useKeepAwake } from 'expo-keep-awake';

import { Glass } from '@/components/glass';
import { PlayMeter } from '@/components/play-meter';

import DrawingCanvas, { type Stroke } from '@/components/drawing-canvas';
import { BrushColors, Palette, Radius, Space, Typography } from '@/constants/design';
import { MODES, type Mode, type RoundResult } from '@/lib/modes';
import { Session, type Snapshot } from '@/lib/session';
import { appendRecord } from '@/lib/history';
import { useDoodleInfer, useDoodleModel } from '@/hooks/use-doodle-ai';

type Run = { id: number; target: number; session: Session };

export default function PlayScreen() {
  useKeepAwake();
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const params = useLocalSearchParams<{ mode?: string }>();
  const mode: Mode = (['challenge', 'blind', 'free'] as const).includes(params.mode as Mode)
    ? (params.mode as Mode)
    : 'challenge';

  const spec = MODES[mode];
  const { model, error, classes } = useDoodleModel();
  const { run, isBlank } = useDoodleInfer(model);
  const nClasses = classes.length || 1;

  // 当前连赢局数。跨局存活，所以放在页面上而不是 Session 里
  // （Session 一局结束就死，连击不该跟着一起死）。
  const [streak, setStreak] = useState(0);

  /**
   * 造一局。**必须在事件处理器里造**，不能在监听 target 的 effect 里派生——
   * 那会造成「渲染 → effect setState → 再渲染」的多余往返，
   * 而且状态机的出生点就该是一个明确的用户动作（点进来、点下一局）。
   *
   * 连击数要顺手交进去：计分公式里有它（每连胜一局加成 15%，最多 5 层），
   * 而计分发生在 Session 内部。早先这里传的是常量 0，等于连击加成从来没生效过。
   */
  const makeSession = useCallback(
    (prev?: number): Run => {
      let t = -1;
      if (spec.needsTarget) {
        t = Math.floor(Math.random() * nClasses);
        if (prev !== undefined && prev >= 0 && nClasses > 1 && t === prev) t = (t + 1) % nClasses;
      }
      return { id: Math.random(), target: t, session: new Session(mode, t, nClasses, streak) };
    },
    [mode, nClasses, spec.needsTarget, streak],
  );

  const [current, setCurrent] = useState<Run>(() => makeSession());
  const { target, session } = current;

  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [color, setColor] = useState<string>(BrushColors[0]);
  const [snap, setSnap] = useState<Snapshot>(() => session.snapshot(Date.now()));
  const [inferMs, setInferMs] = useState(0);
  const [ink, setInk] = useState(0);
  /**
   * 已经落进画面的笔数（含**正在画的那一笔**）。
   *
   * 不能用 `strokes.length`：抬手才 setStrokes，所以整条笔画的过程中它一直不变，
   * 判读区会一直显示「还没落笔，它只能瞎猜」——明明手指已经在画了。
   * 这个值在 applyResult 里跟着推理一起更新（节流后的），频率天然受控。
   */
  const [liveCount, setLiveCount] = useState(0);
  const [result, setResult] = useState<RoundResult | null>(null);
  const [hint, setHint] = useState<string | null>(null);
  const [board, setBoard] = useState(0);

  // 结算只允许发生一次：抬手判定和超时判定有可能在同一毫秒撞上
  const [settled, setSettled] = useState(false);

  /**
   * 落盘 + 触感反馈。
   *
   * `drawn` 必须由调用方传进来，**不能读闭包里的 `strokes`**：
   * 抬手那一刻的执行顺序是 `setStrokes(next)` → `applyResult(next)`，
   * 而 `applyResult` 是上一次渲染时创建的闭包，它的 `strokes` 还停在
   * **少一笔**的旧值上。赢了的那一笔恰好就是最后这一笔——所以闭包版会让
   * 画册里的缩略图永远缺掉决定性的一笔，而且没有任何报错。
   */
  const finish = useCallback(
    (r: RoundResult, drawn: Stroke[]) => {
      setResult(r);
      setSettled(true);
      // 连击只统计「赢」，且自由模式没有输赢、不该打断连击链
      if (r.outcome === 'win') setStreak((n) => n + 1);
      else if (r.outcome === 'lose') setStreak(0);
      const c = target >= 0 ? classes[target] : null;
      void appendRecord({
        mode: r.mode,
        outcome: r.outcome,
        target,
        targetZh: c?.zh ?? '自由画',
        targetEn: c?.en ?? 'free draw',
        emoji: c?.emoji ?? '✏️',
        elapsedMs: r.elapsedMs,
        score: r.score,
        strokeCount: drawn.length,
        ink: session.ink,
        strokes: drawn,
      });
      if (r.outcome === 'win') {
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      } else if (r.outcome === 'lose') {
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      } else {
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
      }
    },
    [classes, target, session],
  );

  // 计时循环需要读最新的 finish / settled / strokes，但不该因为它们变化就重开定时器
  // （strokes 每画一笔都会变）。用一个「最新值盒子」转交：
  // 盒子在 effect 里同步（不是渲染期），定时器回调里读——
  // 这两处都不在渲染期，读 ref 完全合法。
  const liveRef = useRef<{
    finish: (r: RoundResult, drawn: Stroke[]) => void;
    settled: boolean;
    strokes: Stroke[];
  }>({ finish, settled, strokes: [] });
  useEffect(() => {
    liveRef.current = { finish, settled, strokes };
  });

  useEffect(() => {
    const s = current.session;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = () => {
      if (!alive) return;

      // 计时是从**玩家落第一笔**才开始的（见 onStrokeStart）。在那之前状态一直是
      // 'ready'，此时不该推进、也不该 30fps 重渲染整个页面——只是用慢速定时器
      // 保持循环活着，等落笔把状态推到 'playing'。
      //
      // 早先这里写成「只有 'playing' 才排下一次」，第一次 tick 撞上 'ready' 就直接
      // 不排了，定时器当场死掉、再没人救活：倒计时冻结在 60，整局永不超时。
      // 没有报错、没有警告，只是"时间不走"——这类 bug 只能靠读代码抓。
      if (s.state === 'ready') {
        timer = setTimeout(tick, 120);
        return;
      }

      const now = Date.now();
      const r = s.tick(now);
      const st = s.snapshot(now);
      setSnap(st);
      if (r && !liveRef.current.settled) {
        liveRef.current.finish(r, liveRef.current.strokes);
        return;
      }
      if (s.state === 'playing' && spec.durationMs > 0) timer = setTimeout(tick, 33);
    };
    // 第一次 tick 也走 setTimeout：在 effect 体内同步 setState 会立刻多渲染一帧
    if (spec.durationMs > 0) timer = setTimeout(tick, 33);
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [current.session, spec.durationMs]);

  /** 计时从**玩家落第一笔**开始。否则光看题想要画什么，十几秒就没了。 */
  const onStrokeStart = useCallback(() => {
    if (session.state === 'ready') session.start(Date.now());
  }, [session]);

  const applyResult = useCallback(
    (next: Stroke[], force: boolean) => {
      const res = run(next, force);
      if (!res) return;
      setInferMs(res.inferMs);
      setInk(res.ink);
      setLiveCount(next.length);
      if (isBlank(res.ink)) {
        // 擦干净了：把跟踪器清空，别让上一笔的残留判断留在候选栏
        // （阈值统一走 hook 导出的 isBlank，避免这里再写一个魔数）
        session.tracker.reset();
        setSnap(session.snapshot(Date.now()));
        return;
      }
      const now = Date.now();
      session.start(now);
      const r = session.feed(res.probs, now, res.ink);
      setSnap(session.snapshot(now));
      // 把**这一笔已经算进去的** next 传下去，而不是闭包里那份缺一笔的 strokes
      if (r) finish(r, next);
    },
    [run, session, finish, isBlank],
  );

  const onStrokeEnd = useCallback(
    (next: Stroke[]) => {
      setStrokes(next);
      applyResult(next, true);
    },
    [applyResult],
  );

  const onPoint = useCallback(
    (next: Stroke[]) => {
      // 早先这里是 `if (session.state !== 'playing') return;`——但那意味着
      // **第一笔从头到尾都不会被推理**：落笔时状态还是 'ready'，而把它推到
      // 'playing' 的 start() 直到抬手那一次 applyResult 里才调用。
      // 于是"跟着手指走"只在第二笔以后才成立，第一笔画得再久候选栏都是空的。
      // 现在落笔就开局，边画边判从第一个点开始。
      if (session.state === 'ready') session.start(Date.now());
      if (session.state !== 'playing') return;
      applyResult(next, false);
    },
    [applyResult, session],
  );

  const undo = useCallback(() => {
    if (strokes.length === 0) return;
    const next = strokes.slice(0, -1);
    setStrokes(next);
    applyResult(next, true);
  }, [strokes, applyResult]);

  const clearAll = useCallback(() => {
    setStrokes([]);
    setInk(0);
    session.tracker.reset();
    setSnap(session.snapshot(Date.now()));
  }, [session]);

  /**
   * 回上一页。深链直达 `/play` 时（或者 web 上刷新过页面）导航栈是空的，
   * `router.back()` 会抛 `The action 'GO_BACK' was not handled by any navigator`。
   * 那时候退到首页才是玩家真正想要的。
   */
  const goBack = useCallback(() => {
    if (router.canGoBack()) router.back();
    else router.replace('/(tabs)/(home)/index');
  }, [router]);

  const giveUp = useCallback(() => {
    if (settled) return;
    const r = session.giveUp(Date.now());
    if (r) finish(r, strokes);
    else {
      // 自由模式没有输赢，直接退回首页
      goBack();
    }
  }, [session, settled, strokes, finish, goBack]);

  const requestHint = useCallback(() => {
    setHint(session.useHint());
    void Haptics.selectionAsync();
  }, [session]);

  const again = useCallback(() => {
    setSettled(false);
    setResult(null);
    setStrokes([]);
    setInk(0);
    setHint(null);
    const ns = makeSession(target);
    setCurrent(ns);
    setSnap(ns.session.snapshot(Date.now()));
  }, [makeSession, target]);

  const guesses = useMemo(() => snap.guesses.filter((g) => g.p > 0.006), [snap.guesses]);
  const targetClass = target >= 0 ? classes[target] : null;
  const secs = Math.ceil(snap.remainMs / 1000);
  const urgent = spec.durationMs > 0 && secs <= 10;
  const playing = session.state === 'playing' || session.state === 'ready';

  /** 正方形画布：取容器宽高的较小值，避免 aspectRatio 在高度不足时溢出 */
  const onBoardLayout = useCallback((e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setBoard(Math.floor(Math.min(width, height)));
  }, []);

  return (
    <View style={styles.root}>
      {/*
        导航栏交给系统。`Stack.Screen` 写在页面里会覆盖布局里的同名 screen，
        这样标题、返回手势、返回按钮的字形（iOS 是 chevron + 「回去」）全部由
        UINavigationBar 画——转场时它会跟着一起滑动，自己用 View 拼的顶栏做不到。
        右侧的倒计时用 `headerRight`，系统会把它当成一个 bar button 处理，
        动态字体、VoiceOver 顺序都会自动正确。
      */}
      <Stack.Screen
        options={{
          title: spec.title,
          headerRight: () => (
            <View style={styles.headerRight}>
              {streak > 1 && <Text style={styles.streakTag}>连胜 {streak}</Text>}
              {spec.durationMs > 0 ? (
                <Text style={[styles.timer, urgent && styles.timerUrgent]}>{secs}</Text>
              ) : (
                <Text style={styles.freeTag}>自由</Text>
              )}
            </View>
          ),
        }}
      />

      {/* 题目卡 */}
      <View style={[styles.promptCard, mode === 'blind' && styles.promptCardBlind]}>
        <Text style={[styles.promptLabel, mode === 'blind' && styles.promptLabelBlind]}>
          {mode === 'blind' ? '别让它说出来' : mode === 'free' ? '随便画' : '画这个'}
        </Text>
        <View style={styles.promptRow}>
          {targetClass ? (
            <>
              <Text style={styles.promptEmoji}>{targetClass.emoji}</Text>
              <Text style={[styles.promptWord, mode === 'blind' && styles.promptWordBlind]}>
                {targetClass.zh}
              </Text>
            </>
          ) : (
            <Text style={[styles.promptWord, mode === 'blind' && styles.promptWordBlind]}>✏️</Text>
          )}
        </View>
        {mode !== 'free' && (
          <Pressable
            onPress={requestHint}
            hitSlop={10}
            style={[styles.hintBtn, mode === 'blind' && styles.hintBtnBlind]}>
            <Text style={[styles.hintBtnText, mode === 'blind' && { color: Palette.paper }]}>提示</Text>
          </Pressable>
        )}
      </View>

      {/*
        判读区。原先这里是画布上方的空带——竖屏 + 正方形画布的必然产物
        （画布宽度被屏幕卡死，高度就只能等于宽度，多出来的高度没处放）。
        与其让它空着，不如做成这局最该看的读数。

        抽成 `PlayMeter` 并 `memo` 的原因写在那个文件里：计时器每 33ms 重渲染整页，
        而置信度条在 iOS 上是真 SwiftUI `Gauge`，跨边界调用要钱。
      */}
      <PlayMeter
        mode={mode}
        pct={Math.round(snap.closeness * 100)}
        rank={snap.rank}
        strokeCount={liveCount}
        secs={Math.round((snap.elapsedMs || 0) / 1000)}
        ink={Math.round(ink)}
        minInk={spec.minInk}
        hint={hint}
      />

      {/* 画布 */}
      <View style={styles.boardArea} onLayout={onBoardLayout}>
        {board > 0 && (
          <View
            style={[
              styles.board,
              { width: board, height: board },
              snap.danger && mode === 'blind' && styles.boardDanger,
            ]}>
            <DrawingCanvas
              strokes={strokes}
              color={color}
              enabled={!settled && !!model}
              onStrokeStart={onStrokeStart}
              onStrokeEnd={onStrokeEnd}
              onPoint={onPoint}
            />
          </View>
        )}
        {!model && !error && <Text style={styles.hintOverlay}>正在唤醒模型…</Text>}
        {error && <Text style={styles.errorOverlay}>模型加载失败：{error}</Text>}
      </View>

      {/* 判定进度条：挑战模式是"快通过了"，盲区模式是"快被看穿了"，颜色相反 */}
      <View style={styles.holdTrack}>
        <View
          style={[
            styles.holdFill,
            mode === 'blind' && { backgroundColor: Palette.seal },
            { width: `${Math.round(snap.holdRatio * 100)}%` },
          ]}
        />
      </View>

      {/* 候选栏：毛玻璃。iOS 26 上是系统 Liquid Glass，旧系统退回 UIBlurEffect */}
      <View style={styles.guessWrap}>
        <Glass style={StyleSheet.absoluteFill} />
        <View style={styles.guessHeader}>
          <Text style={styles.guessTitle}>
            {mode === 'blind' ? '它在想（别让它中）' : '它在想'}
          </Text>
          <Text style={styles.guessMeta}>
            {inferMs > 0 ? `${inferMs}ms` : '—'}
            {/* 没落笔时追踪器全零，rank 只是个任意的序号，报出来会误导 */}
            {mode !== 'free' && snap.rank >= 0 && strokes.length > 0 ? ` · 第${snap.rank + 1}位` : ''}
          </Text>
        </View>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.guessRow}>
          {guesses.length === 0 ? (
            <Text style={styles.guessEmpty}>
              {/* 跟踪器为空 = 真的还没落笔；跟踪器有值但全被 0.006 的显示阈值滤掉
                  = 画了但没一个类占到边。这两句话对玩家是不同信息，不能混用。 */}
              {snap.guesses.length === 0 ? '动笔，它就开始猜了' : '它在看，但还说不准'}
            </Text>
          ) : (
            guesses.map((g, i) => {
              const c = classes[g.index];
              const isTarget = g.index === target;
              return (
                <View
                  key={g.index}
                  style={[
                    styles.chip,
                    isTarget && styles.chipTarget,
                    mode === 'blind' && isTarget && styles.chipDanger,
                  ]}>
                  <Text style={styles.chipRank}>{i + 1}</Text>
                  <Text style={styles.chipEmoji}>{c?.emoji}</Text>
                  <Text
                    style={[
                      styles.chipWord,
                      isTarget && styles.chipWordTarget,
                      // 盲区模式的目标 chip 是朱红实底，字必须转白，否则朱红压朱红看不见
                      mode === 'blind' && isTarget && styles.chipWordDanger,
                    ]}
                    numberOfLines={1}>
                    {c?.zh}
                  </Text>
                  <Text
                    style={[styles.chipPct, mode === 'blind' && isTarget && styles.chipPctDanger]}>
                    {Math.round(g.p * 100)}%
                  </Text>
                </View>
              );
            })
          )}
        </ScrollView>
      </View>

      {/* 工具栏 */}
      <View style={[styles.toolbar, { paddingBottom: insets.bottom + Space.sm }]}>
        <View style={styles.colors}>
          {BrushColors.map((c) => (
            <Pressable
              key={c}
              onPress={() => setColor(c)}
              hitSlop={4}
              style={[
                styles.swatch,
                { backgroundColor: c },
                color === c && styles.swatchOn,
              ]}
            />
          ))}
        </View>
        <View style={{ flex: 1 }} />
        <ToolButton label="撤销" onPress={undo} disabled={strokes.length === 0 || !playing} />
        <ToolButton label="清空" onPress={clearAll} disabled={strokes.length === 0 || !playing} />
        <ToolButton label={mode === 'free' ? '完成' : '放弃'} onPress={giveUp} disabled={!playing} />
      </View>

      {result && (
        <ResultCard
          result={result}
          word={targetClass ? `${targetClass.emoji} ${targetClass.zh}` : '自由画'}
          onAgain={again}
          onHome={goBack}
        />
      )}
    </View>
  );
}

function ToolButton({
  label,
  onPress,
  disabled,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
}) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [
        styles.toolBtn,
        pressed && !disabled && styles.toolBtnPressed,
      ]}>
      <Text style={[styles.toolBtnText, disabled && { color: Palette.inkFaint }]}>{label}</Text>
    </Pressable>
  );
}

function ResultCard({
  result,
  word,
  onAgain,
  onHome,
}: {
  result: RoundResult;
  word: string;
  onAgain: () => void;
  onHome: () => void;
}) {
  // Animated.Value 一旦创建就该固定。用 useState 惰性初始化持有它；
  // useRef(...).current 会在渲染期读 ref，React 19 会直接报错。
  const [scale] = useState(() => new Animated.Value(0.88));
  const [fade] = useState(() => new Animated.Value(0));

  useEffect(() => {
    Animated.parallel([
      Animated.spring(scale, { toValue: 1, useNativeDriver: true, friction: 7, tension: 78 }),
      Animated.timing(fade, {
        toValue: 1,
        duration: 170,
        useNativeDriver: true,
        easing: Easing.out(Easing.quad),
      }),
    ]).start();
  }, [scale, fade]);

  const glyph = result.outcome === 'win' ? '✓' : result.outcome === 'lose' ? '✕' : '·';
  const accent =
    result.outcome === 'win' ? Palette.jade : result.outcome === 'lose' ? Palette.seal : Palette.inkFaint;

  return (
    <View style={styles.overlay}>
      <Animated.View style={[styles.resultCard, { opacity: fade, transform: [{ scale }] }]}>
        <View style={[styles.resultBadge, { backgroundColor: accent }]}>
          <Text style={styles.resultBadgeGlyph}>{glyph}</Text>
        </View>
        <Text style={styles.resultTitle}>{result.headline}</Text>
        <Text style={styles.resultWord}>{word}</Text>
        <Text style={styles.resultDetail}>{result.detail}</Text>
        {result.score > 0 && <Text style={styles.resultScore}>+{result.score}</Text>}
        <View style={styles.resultBtns}>
          <Pressable
            onPress={onHome}
            style={({ pressed }) => [styles.ghostBtn, pressed && { opacity: 0.7 }]}>
            <Text style={styles.ghostBtnText}>回去</Text>
          </Pressable>
          <Pressable
            onPress={onAgain}
            style={({ pressed }) => [styles.solidBtn, pressed && { opacity: 0.85 }]}>
            <Text style={styles.solidBtnText}>再来一局</Text>
          </Pressable>
        </View>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Palette.paper, paddingHorizontal: Space.lg },

  // 导航栏右侧：倒计时与连击。系统 bar button 只有 44pt 高，元素都压得很小
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  streakTag: {
    color: Palette.seal,
    fontSize: 10.5,
    fontFamily: Typography.zh,
    fontWeight: '700',
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: Radius.pill,
    backgroundColor: Palette.sealWash,
    overflow: 'hidden',
  },
  timer: { color: Palette.ink, fontSize: 24, fontWeight: '700', fontFamily: Typography.mono },
  timerUrgent: { color: Palette.seal },
  freeTag: { color: Palette.inkFaint, fontSize: 11, fontFamily: Typography.zh },

  promptCard: {
    marginTop: Space.sm,
    backgroundColor: Palette.sealWash,
    borderRadius: Radius.md,
    paddingVertical: Space.sm,
    paddingHorizontal: Space.lg,
    flexDirection: 'row',
    alignItems: 'center',
    gap: Space.sm,
  },
  promptCardBlind: { backgroundColor: Palette.ink },
  promptLabel: { color: Palette.seal, fontSize: 11, fontFamily: Typography.zh, opacity: 0.8 },
  promptLabelBlind: { color: '#F5B3A6' },
  promptRow: { flexDirection: 'row', alignItems: 'center', gap: Space.sm, flex: 1 },
  promptEmoji: { fontSize: 21 },
  promptWord: { color: Palette.ink, fontSize: 23, fontWeight: '700', fontFamily: Typography.zh },
  promptWordBlind: { color: Palette.paper },
  hintBtn: {
    paddingHorizontal: Space.md,
    paddingVertical: 5,
    borderRadius: Radius.pill,
    backgroundColor: Palette.card,
    borderWidth: 1,
    borderColor: Palette.seal,
  },
  hintBtnBlind: { backgroundColor: 'transparent', borderColor: 'rgba(251,250,246,0.45)' },
  hintBtnText: { color: Palette.seal, fontSize: 11.5, fontFamily: Typography.zh, fontWeight: '600' },

  boardArea: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: Space.sm,
  },
  board: {
    borderRadius: Radius.lg,
    backgroundColor: Palette.card,
    borderWidth: 1,
    borderColor: Palette.rule,
    overflow: 'hidden',
  },
  boardDanger: { borderColor: Palette.seal, borderWidth: 2 },
  hintOverlay: {
    position: 'absolute',
    color: Palette.inkFaint,
    fontSize: 13,
    fontFamily: Typography.zh,
  },
  errorOverlay: {
    position: 'absolute',
    color: Palette.seal,
    fontSize: 12,
    fontFamily: Typography.mono,
    paddingHorizontal: 24,
    textAlign: 'center',
  },

  holdTrack: {
    height: 3,
    borderRadius: 2,
    backgroundColor: Palette.paperSunken,
    overflow: 'hidden',
  },
  holdFill: { height: 3, backgroundColor: Palette.jade },

  guessWrap: {
    marginTop: Space.sm,
    borderRadius: Radius.md,
    overflow: 'hidden',
    backgroundColor: 'rgba(255,255,255,0.6)',
    paddingBottom: Space.xs,
  },
  guessHeader: {
    flexDirection: 'row',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    paddingHorizontal: Space.md,
    paddingTop: Space.sm,
  },
  guessTitle: { color: Palette.inkSoft, fontSize: 11.5, fontFamily: Typography.zh, fontWeight: '600' },
  guessMeta: { color: Palette.inkFaint, fontSize: 10.5, fontFamily: Typography.mono },
  guessRow: { gap: 7, paddingHorizontal: Space.md, paddingVertical: Space.sm, alignItems: 'center' },
  guessEmpty: { color: Palette.inkFaint, fontSize: 11.5, fontFamily: Typography.zh, paddingVertical: 12 },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 9,
    paddingVertical: 5,
    borderRadius: Radius.pill,
    backgroundColor: 'rgba(255,255,255,0.72)',
    borderWidth: 1,
    borderColor: 'rgba(230,227,218,0.9)',
    maxWidth: 140,
  },
  chipTarget: { borderColor: Palette.seal, backgroundColor: Palette.sealWash },
  chipDanger: { borderColor: Palette.seal, backgroundColor: Palette.seal },
  chipRank: { color: Palette.inkFaint, fontSize: 9.5, fontFamily: Typography.mono },
  chipEmoji: { fontSize: 12.5 },
  chipWord: { color: Palette.ink, fontSize: 12.5, fontFamily: Typography.zh, flexShrink: 1 },
  chipWordTarget: { color: Palette.seal, fontWeight: '700' },
  chipWordDanger: { color: Palette.paper },
  chipPctDanger: { color: 'rgba(251,250,246,0.7)' },
  chipPct: { color: Palette.inkFaint, fontSize: 9.5, fontFamily: Typography.mono },

  toolbar: { flexDirection: 'row', alignItems: 'center', gap: Space.sm, paddingTop: Space.sm },
  colors: { flexDirection: 'row', gap: 6, alignItems: 'center' },
  swatch: { width: 20, height: 20, borderRadius: 10 },
  swatchOn: { borderWidth: 2.5, borderColor: Palette.ink },
  toolBtn: {
    paddingHorizontal: Space.md,
    paddingVertical: 7,
    borderRadius: Radius.sm,
    backgroundColor: Palette.paperSunken,
  },
  toolBtnPressed: { backgroundColor: Palette.rule },
  toolBtnText: { color: Palette.inkSoft, fontSize: 11.5, fontFamily: Typography.zh, fontWeight: '600' },

  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(23,22,26,0.4)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  resultCard: {
    width: '84%',
    backgroundColor: Palette.card,
    borderRadius: Radius.lg,
    padding: Space.xl,
    alignItems: 'center',
    gap: 5,
  },
  resultBadge: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 2,
  },
  resultBadgeGlyph: { color: '#FFF', fontSize: 22, fontWeight: '700' },
  resultTitle: { color: Palette.ink, fontSize: 21, fontWeight: '700', fontFamily: Typography.zh },
  resultWord: { color: Palette.inkSoft, fontSize: 15, fontFamily: Typography.zh },
  resultDetail: {
    color: Palette.inkFaint,
    fontSize: 12,
    fontFamily: Typography.zh,
    textAlign: 'center',
    lineHeight: 18,
    marginTop: 2,
  },
  resultScore: { color: Palette.seal, fontSize: 30, fontWeight: '700', fontFamily: Typography.mono },
  resultBtns: { flexDirection: 'row', gap: Space.sm, marginTop: Space.md, width: '100%' },
  ghostBtn: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 12,
    borderRadius: Radius.md,
    backgroundColor: Palette.paperSunken,
  },
  ghostBtnText: { color: Palette.inkSoft, fontSize: 14, fontWeight: '600', fontFamily: Typography.zh },
  solidBtn: {
    flex: 2,
    alignItems: 'center',
    paddingVertical: 12,
    borderRadius: Radius.md,
    backgroundColor: Palette.ink,
  },
  solidBtnText: { color: Palette.paper, fontSize: 14, fontWeight: '700', fontFamily: Typography.zh },
});
