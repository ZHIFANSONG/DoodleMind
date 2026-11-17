/**
 * 游戏主界面。
 *
 * 屏幕从上到下是三层信息密度递减的区块：
 *   1. 顶部窄条 —— 只放「还剩几秒」和连击，扫一眼就知道局势
 *   2. 题目卡 —— 目标词 + emoji，朱红底让它成为视觉焦点
 *   3. 画布 —— 占掉剩余全部空间，越大的画布模型越准
 * 底部是 AI 的实时候选栏，这是整个游戏的信息核心：
 * 玩家不是在跟人猜，是在观察一个黑箱的注意力往哪偏。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Animated, Easing, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import * as Haptics from 'expo-haptics';
import { useKeepAwake } from 'expo-keep-awake';

import DrawingCanvas, { type Stroke } from '@/components/drawing-canvas';
import { BrushColors, Palette, Radius, Space, Typography } from '@/constants/design';
import { DEFAULT_ROUND, Round, hintFor, scoreRound, type RoundSnapshot } from '@/lib/game';
import { appendRecord } from '@/lib/history';
import { useDoodleInfer, useDoodleModel } from '@/hooks/use-doodle-ai';

const ROUND_LIMIT = DEFAULT_ROUND.durationMs;
const TOP_N = 6;

type Session = { id: number; target: number; round: Round };

export default function GameScreen() {
  useKeepAwake();
  const insets = useSafeAreaInsets();
  const { model, error, classes } = useDoodleModel();
  const { run } = useDoodleInfer(model);

  const nClasses = classes.length || 1;

  /**
   * 造一局：随机目标词 + 全新状态机。
   *
   * 关键点：局是在**事件处理器里**造出来的（开局一次、点「下一题」一次），
   * 不是在一个监听 target 变化的 effect 里。用 effect 派生状态会导致
   * 「渲染 → effect 里 setState → 再渲染」的多余往返，而且 React 19 的
   * lint 会直接判定这是反模式。状态机的出生点必须是一个明确的事件。
   */
  const makeSession = useCallback(
    (prev?: number): Session => {
      let t = Math.floor(Math.random() * nClasses);
      if (prev !== undefined && nClasses > 1 && t === prev) t = (t + 1) % nClasses;
      return { id: Math.random(), target: t, round: new Round(t, nClasses, DEFAULT_ROUND) };
    },
    [nClasses],
  );

  const [session, setSession] = useState<Session>(() => makeSession());
  const { target, round } = session;

  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [color, setColor] = useState<string>(BrushColors[0]);
  const [snap, setSnap] = useState<RoundSnapshot>(() => round.snapshot(Date.now()));
  const [inferMs, setInferMs] = useState(0);
  const [streak, setStreak] = useState(0);
  const [totalScore, setTotalScore] = useState(0);
  const [hint, setHint] = useState<string | null>(null);
  const [ended, setEnded] = useState<null | { cleared: boolean; score: number; elapsedMs: number }>(null);

  /** 防止同一局被结算两次（抬手判定和超时判定可能撞在一起） */
  const savedRef = useRef(false);

  // 结算：先把这一局落盘，再动连击
  const finish = useCallback(
    (cleared: boolean, s: RoundSnapshot) => {
      if (savedRef.current) return;
      savedRef.current = true;
      const used = s.elapsedMs;
      const score = cleared
        ? scoreRound({ elapsedMs: used, hintsUsed: round.hintsUsed, streakBefore: streak })
        : 0;
      setEnded({ cleared, score, elapsedMs: used });
      setStreak((v) => (cleared ? v + 1 : 0));
      if (cleared) setTotalScore((v) => v + score);
      void Haptics.notificationAsync(
        cleared ? Haptics.NotificationFeedbackType.Success : Haptics.NotificationFeedbackType.Warning,
      );
      const c = classes[target] ?? { en: '', zh: '', emoji: '' };
      void appendRecord({
        target,
        targetZh: c.zh,
        targetEn: c.en,
        emoji: c.emoji,
        cleared,
        elapsedMs: used,
        score,
        strokeCount: strokes.length,
        strokes,
      });
    },
    [classes, streak, strokes, target, round],
  );

  /** 计时循环需要读最新的 finish，但不能因为它每次渲染都变就把定时器重开一遍 */
  const finishRef = useRef(finish);
  useEffect(() => {
    finishRef.current = finish;
  });

  // 计时循环：30fps 足够，60fps 纯属浪费电。
  // 第一次 tick 也走 setTimeout——在 effect 体内同步跑会立刻 setState，
  // 白白多渲染一帧，而且会让 lint 判为同步副作用。
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = () => {
      if (!alive) return;
      const now = Date.now();
      round.tick(now);
      const s = round.snapshot(now);
      setSnap(s);
      if (s.state === 'timeout') {
        finishRef.current(false, s);
        return;
      }
      timer = setTimeout(tick, 33);
    };
    timer = setTimeout(tick, 33);
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [round]);

  /**
   * 计时从**玩家落第一笔**开始，不是从进页面开始。
   * 否则光是看清楚题目、想想画什么，十几秒就没了，很劝退。
   */
  const onStrokeStart = useCallback(() => {
    if (round.state === 'ready') round.start(Date.now());
  }, [round]);

  const onStrokeEnd = useCallback(
    (next: Stroke[]) => {
      setStrokes(next);
      if (round.state === 'ready') round.start(Date.now());
      const res = run(next, true);
      if (!res) return;
      setInferMs(res.inferMs);
      const passed = round.feed(res.probs, Date.now());
      const s = round.snapshot(Date.now());
      setSnap(s);
      if (passed) {
        void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
        finish(true, s);
      }
    },
    [run, finish, round],
  );

  const onPoint = useCallback(
    (next: Stroke[]) => {
      if (round.state !== 'playing') return;
      const res = run(next, false);
      if (!res) return;
      setInferMs(res.inferMs);
      round.feed(res.probs, Date.now());
      setSnap(round.snapshot(Date.now()));
    },
    [run, round],
  );

  const undo = useCallback(() => {
    setStrokes((prev) => {
      const next = prev.slice(0, -1);
      // 撤销后立刻重跑，否则候选栏会停在旧判断上
      setTimeout(() => {
        if (round.state !== 'playing') return;
        const res = run(next, true);
        if (!res) return;
        round.feed(res.probs, Date.now());
        setSnap(round.snapshot(Date.now()));
      }, 0);
      return next;
    });
  }, [run, round]);

  const clearAll = useCallback(() => {
    setStrokes([]);
    setSnap((s) => ({ ...s, closeness: 0, rank: 99, guesses: [], holdRatio: 0 }));
  }, []);

  const requestHint = useCallback(() => {
    if (round.state !== 'playing') return;
    round.hintsUsed += 1;
    setHint(hintFor(snap.rank, snap.closeness));
    void Haptics.selectionAsync();
  }, [round, snap]);

  const next = useCallback(() => {
    const ns = makeSession(target);
    savedRef.current = false;
    setSession(ns);
    setStrokes([]);
    setHint(null);
    setEnded(null);
    setSnap(ns.round.snapshot(Date.now()));
  }, [makeSession, target]);

  const skip = useCallback(() => {
    if (round.state !== 'playing') return;
    round.skip();
    finish(false, round.snapshot(Date.now()));
  }, [finish, round]);

  const remain = snap.remainMs;
  const secs = Math.ceil(remain / 1000);
  const urgent = secs <= 10;

  const guesses = useMemo(() => snap.guesses.filter((x) => x.p > 0.008).slice(0, TOP_N), [snap]);

  const targetClass = classes[target];
  const holdRatio = snap.holdRatio;

  return (
    <View style={[styles.root, { paddingTop: insets.top + Space.sm }]}>
      <View style={styles.topBar}>
        <View style={styles.statChip}>
          <Text style={styles.statLabel}>连击</Text>
          <Text style={styles.statValue}>{streak}</Text>
        </View>
        <View style={styles.statChip}>
          <Text style={styles.statLabel}>总分</Text>
          <Text style={styles.statValue}>{totalScore}</Text>
        </View>
        <View style={{ flex: 1 }} />
        <Text style={[styles.timer, urgent && styles.timerUrgent]}>{secs}s</Text>
      </View>

      <View style={styles.promptCard}>
        <Text style={styles.promptLabel}>画这个</Text>
        <View style={styles.promptRow}>
          <Text style={styles.promptEmoji}>{targetClass?.emoji}</Text>
          <Text style={styles.promptWord}>{targetClass?.zh}</Text>
        </View>
        <Pressable onPress={requestHint} hitSlop={10} style={styles.hintBtn}>
          <Text style={styles.hintBtnText}>提示</Text>
        </Pressable>
      </View>

      {hint && <Text style={styles.hintText}>{hint}</Text>}

      <View style={styles.canvasWrap}>
        <DrawingCanvas
          strokes={strokes}
          color={color}
          enabled={!ended && !!model}
          onStrokeStart={onStrokeStart}
          onStrokeEnd={onStrokeEnd}
          onPoint={onPoint}
        />
        {error && <Text style={styles.errorText}>模型加载失败：{error}</Text>}
        {!model && !error && <Text style={styles.loadingText}>正在唤醒模型…</Text>}
      </View>

      <View style={styles.holdTrack}>
        <View style={[styles.holdFill, { width: `${Math.round(holdRatio * 100)}%` }]} />
      </View>

      <View style={styles.guessWrap}>
        <View style={styles.guessHeader}>
          <Text style={styles.guessTitle}>它在想</Text>
          <Text style={styles.guessMeta}>
            {inferMs > 0 ? `${inferMs}ms` : '—'} · 第{snap.rank + 1}位
          </Text>
        </View>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.guessRow}>
          {guesses.length === 0 && <Text style={styles.guessEmpty}>动笔，它就会开始猜</Text>}
          {guesses.map((g, i) => {
            const c = classes[g.index];
            const isTarget = g.index === target;
            return (
              <View key={g.index} style={[styles.guessChip, isTarget && styles.guessChipTarget]}>
                <Text style={styles.guessRank}>{i + 1}</Text>
                <Text style={styles.guessEmoji}>{c?.emoji}</Text>
                <Text style={[styles.guessWord, isTarget && styles.guessWordTarget]} numberOfLines={1}>
                  {c?.zh}
                </Text>
                <Text style={styles.guessPct}>{Math.round(g.p * 100)}%</Text>
              </View>
            );
          })}
        </ScrollView>
      </View>

      <View style={[styles.toolbar, { paddingBottom: insets.bottom + Space.sm }]}>
        <View style={styles.colors}>
          {BrushColors.map((c) => (
            <Pressable
              key={c}
              onPress={() => setColor(c)}
              style={[
                styles.swatch,
                { backgroundColor: c },
                color === c && { borderColor: Palette.ink, borderWidth: 2.5 },
              ]}
            />
          ))}
        </View>
        <View style={{ flex: 1 }} />
        <ToolButton label="撤销" onPress={undo} disabled={strokes.length === 0 || !!ended} />
        <ToolButton label="清空" onPress={clearAll} disabled={strokes.length === 0 || !!ended} />
        <ToolButton label="跳过" onPress={skip} disabled={!!ended} />
      </View>

      {ended && (
        <ResultOverlay
          cleared={ended.cleared}
          score={ended.score}
          elapsedMs={ended.elapsedMs}
          word={targetClass?.zh ?? ''}
          emoji={targetClass?.emoji ?? ''}
          onNext={next}
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
        pressed && styles.toolBtnPressed,
        disabled && styles.toolBtnDisabled,
      ]}>
      <Text style={[styles.toolBtnText, disabled && { color: Palette.inkFaint }]}>{label}</Text>
    </Pressable>
  );
}

function ResultOverlay({
  cleared,
  score,
  elapsedMs,
  word,
  emoji,
  onNext,
}: {
  cleared: boolean;
  score: number;
  elapsedMs: number;
  word: string;
  emoji: string;
  onNext: () => void;
}) {
  const scale = useRef(new Animated.Value(0.86)).current;
  const fade = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.spring(scale, { toValue: 1, useNativeDriver: true, friction: 7, tension: 70 }),
      Animated.timing(fade, {
        toValue: 1,
        duration: 180,
        useNativeDriver: true,
        easing: Easing.out(Easing.quad),
      }),
    ]).start();
  }, [scale, fade]);

  return (
    <View style={styles.overlay}>
      <Animated.View style={[styles.resultCard, { opacity: fade, transform: [{ scale }] }]}>
        <Text style={styles.resultEmoji}>{cleared ? '🎉' : '⏳'}</Text>
        <Text style={styles.resultTitle}>{cleared ? '它猜到了' : '时间到'}</Text>
        <Text style={styles.resultWord}>
          {emoji} {word}
        </Text>
        {cleared ? (
          <>
            <Text style={styles.resultScore}>+{score}</Text>
            <Text style={styles.resultMeta}>用时 {(elapsedMs / 1000).toFixed(1)}s</Text>
          </>
        ) : (
          <Text style={styles.resultMeta}>下次画大一点、线条少一点</Text>
        )}
        <Pressable
          onPress={onNext}
          style={({ pressed }) => [styles.nextBtn, pressed && { opacity: 0.8 }]}>
          <Text style={styles.nextBtnText}>下一题</Text>
        </Pressable>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Palette.paper, paddingHorizontal: Space.lg },

  topBar: { flexDirection: 'row', alignItems: 'center', gap: Space.sm },
  statChip: { flexDirection: 'row', alignItems: 'baseline', gap: 5 },
  statLabel: { color: Palette.inkFaint, fontSize: 11, fontFamily: Typography.zh },
  statValue: { color: Palette.ink, fontSize: 15, fontWeight: '700', fontFamily: Typography.mono },
  timer: { color: Palette.inkSoft, fontSize: 26, fontWeight: '700', fontFamily: Typography.mono },
  timerUrgent: { color: Palette.seal },

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
  promptLabel: { color: Palette.seal, fontSize: 12, fontFamily: Typography.zh, opacity: 0.75 },
  promptRow: { flexDirection: 'row', alignItems: 'center', gap: Space.sm, flex: 1 },
  promptEmoji: { fontSize: 22 },
  promptWord: { color: Palette.ink, fontSize: 24, fontWeight: '700', fontFamily: Typography.zh },
  hintBtn: {
    paddingHorizontal: Space.md,
    paddingVertical: 5,
    borderRadius: Radius.pill,
    backgroundColor: Palette.card,
    borderWidth: 1,
    borderColor: Palette.seal,
  },
  hintBtnText: { color: Palette.seal, fontSize: 12, fontFamily: Typography.zh, fontWeight: '600' },
  hintText: {
    marginTop: Space.xs,
    color: Palette.inkSoft,
    fontSize: 12,
    fontFamily: Typography.zh,
    paddingHorizontal: 2,
  },

  canvasWrap: {
    flex: 1,
    marginVertical: Space.sm,
    borderRadius: Radius.lg,
    backgroundColor: Palette.card,
    borderWidth: 1,
    borderColor: Palette.rule,
    overflow: 'hidden',
  },
  loadingText: {
    position: 'absolute',
    alignSelf: 'center',
    top: '48%',
    color: Palette.inkFaint,
    fontSize: 13,
    fontFamily: Typography.zh,
  },
  errorText: {
    position: 'absolute',
    alignSelf: 'center',
    top: '45%',
    color: Palette.seal,
    fontSize: 12,
    fontFamily: Typography.mono,
    paddingHorizontal: 20,
    textAlign: 'center',
  },

  holdTrack: {
    height: 4,
    borderRadius: 2,
    backgroundColor: Palette.paperSunken,
    overflow: 'hidden',
  },
  holdFill: { height: 4, backgroundColor: Palette.jade },

  guessWrap: { paddingTop: Space.sm },
  guessHeader: { flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between' },
  guessTitle: { color: Palette.inkSoft, fontSize: 12, fontFamily: Typography.zh, fontWeight: '600' },
  guessMeta: { color: Palette.inkFaint, fontSize: 11, fontFamily: Typography.mono },
  guessRow: { gap: Space.sm, paddingVertical: Space.sm, alignItems: 'center' },
  guessEmpty: { color: Palette.inkFaint, fontSize: 12, fontFamily: Typography.zh, paddingVertical: 14 },
  guessChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: Space.sm,
    paddingVertical: 6,
    borderRadius: Radius.pill,
    backgroundColor: Palette.card,
    borderWidth: 1,
    borderColor: Palette.rule,
    maxWidth: 150,
  },
  guessChipTarget: { borderColor: Palette.seal, backgroundColor: Palette.sealWash },
  guessRank: { color: Palette.inkFaint, fontSize: 10, fontFamily: Typography.mono },
  guessEmoji: { fontSize: 14 },
  guessWord: { color: Palette.ink, fontSize: 13, fontFamily: Typography.zh, flexShrink: 1 },
  guessWordTarget: { color: Palette.seal, fontWeight: '700' },
  guessPct: { color: Palette.inkFaint, fontSize: 10, fontFamily: Typography.mono },

  toolbar: { flexDirection: 'row', alignItems: 'center', gap: Space.sm, paddingTop: Space.xs },
  colors: { flexDirection: 'row', gap: 7, alignItems: 'center' },
  swatch: { width: 22, height: 22, borderRadius: 11, borderWidth: 0 },
  toolBtn: {
    paddingHorizontal: Space.md,
    paddingVertical: 7,
    borderRadius: Radius.sm,
    backgroundColor: Palette.paperSunken,
  },
  toolBtnPressed: { backgroundColor: Palette.rule },
  toolBtnDisabled: { backgroundColor: Palette.paper },
  toolBtnText: { color: Palette.inkSoft, fontSize: 12, fontFamily: Typography.zh, fontWeight: '600' },

  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(23,22,26,0.42)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  resultCard: {
    width: '82%',
    backgroundColor: Palette.card,
    borderRadius: Radius.lg,
    padding: Space.xl,
    alignItems: 'center',
    gap: 6,
  },
  resultEmoji: { fontSize: 40 },
  resultTitle: { color: Palette.ink, fontSize: 22, fontWeight: '700', fontFamily: Typography.zh },
  resultWord: { color: Palette.inkSoft, fontSize: 17, fontFamily: Typography.zh },
  resultScore: { color: Palette.seal, fontSize: 34, fontWeight: '700', fontFamily: Typography.mono },
  resultMeta: { color: Palette.inkFaint, fontSize: 12, fontFamily: Typography.zh },
  nextBtn: {
    marginTop: Space.md,
    width: '100%',
    alignItems: 'center',
    paddingVertical: 13,
    borderRadius: Radius.md,
    backgroundColor: Palette.ink,
  },
  nextBtnText: { color: Palette.paper, fontSize: 15, fontWeight: '700', fontFamily: Typography.zh },
});

