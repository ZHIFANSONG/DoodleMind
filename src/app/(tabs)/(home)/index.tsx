/**
 * 首页：选玩法。
 *
 * 这一页的存在本身就是个设计决定。如果打开 App 直接进画布，
 * 「盲区」这个玩法就永远没人会发现——它是反向的，规则和常识相反，
 * 必须用一整块卡片把「让它认不出来」讲清楚，玩家才敢点。
 *
 * 所以首页不是菜单，是**三种不同赌注的陈列**：每张卡片写清你赌的是什么、
 * 输赢怎么算。不做图标网格，因为这里卖的是规则本身。
 */

import { useCallback, useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useScreenTop } from '@/hooks/use-screen-top';
import { useFocusEffect, useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';

import { Glyph } from '@/components/glyph';

import { Palette, Radius, Space, Typography } from '@/constants/design';
import { MODES, MODE_ORDER, modeSummary, type Mode } from '@/lib/modes';
import { loadHistory, summarize } from '@/lib/history';
import { useDoodleModel, useModelInfo } from '@/hooks/use-doodle-ai';

export default function HomeScreen() {
  const insets = useSafeAreaInsets();
  const padTop = useScreenTop();
  const router = useRouter();
  const { model, error, loadMs, classes } = useDoodleModel();
  const info = useModelInfo();

  const [stats, setStats] = useState<{ rounds: number; won: number; best: number; winRate: number }>({
    rounds: 0,
    won: 0,
    best: 0,
    winRate: 0,
  });

  // 从游戏页返回时刷新统计，不需要全局状态库
  useFocusEffect(
    useCallback(() => {
      let alive = true;
      void loadHistory().then((list) => {
        if (alive) setStats(summarize(list));
      });
      return () => {
        alive = false;
      };
    }, []),
  );

  const start = useCallback(
    (mode: Mode) => {
      void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      router.push({ pathname: '/play', params: { mode } });
    },
    [router],
  );

  const ready = !!model;
  const statusText = useMemo(() => {
    if (error) return '模型加载失败';
    if (!ready) return '正在唤醒模型…';
    return `${classes.length} 类 · ${loadMs}ms 唤醒`;
  }, [error, ready, classes.length, loadMs]);

  return (
    <ScrollView
      style={styles.root}
      contentContainerStyle={{
        paddingTop: padTop + Space.lg,
        paddingBottom: insets.bottom + Space.xxl,
      }}
      showsVerticalScrollIndicator={false}>
      {/* 品牌区。
          标题「DoodleMind」已经交给 UINavigationBar 的大标题了——系统会在滚动时
          把它缩小吸顶、给背景加毛玻璃。这里只留副标与状态，不重复写第二遍名字。 */}
      <View style={styles.brand}>
        <Text style={styles.brandLine}>
          一个只有 {(info.params / 10000).toFixed(1)} 万参数的小网络，装在你手机里。{'\n'}
          它不会上网，也不知道自己错在哪。
        </Text>
        <View style={[styles.statusPill, error && styles.statusPillError]}>
          <View style={[styles.statusDot, ready && styles.statusDotReady]} />
          <Text style={styles.statusText}>{statusText}</Text>
        </View>
      </View>

      {/* 三种玩法 */}
      <View style={styles.modes}>
        {MODE_ORDER.map((m, i) => {
          const spec = MODES[m];
          return (
            <Pressable
              key={m}
              onPress={() => start(m)}
              disabled={!ready}
              style={({ pressed }) => [
                styles.card,
                i === 0 && styles.cardPrimary,
                pressed && styles.cardPressed,
                !ready && styles.cardDisabled,
              ]}>
              <View style={styles.cardHead}>
                <View style={[styles.glyphBox, i === 0 && styles.glyphBoxPrimary]}>
                  <Glyph
                    name={spec.glyph}
                    color={i === 0 ? Palette.paper : Palette.ink}
                    size={20}
                  />
                </View>
                <View style={{ flex: 1 }}>
                  <Text style={[styles.cardTitle, i === 0 && styles.cardTitlePrimary]}>
                    {spec.title}
                  </Text>
                  <Text style={[styles.cardTagline, i === 0 && styles.cardTaglinePrimary]}>
                    {modeSummary(m)}
                  </Text>
                </View>
                <Glyph name="chevronRight" color={i === 0 ? Palette.paper : Palette.inkFaint} size={14} />
              </View>

              <View style={styles.ruleList}>
                {spec.rules.map((r) => (
                  <View key={r} style={styles.ruleRow}>
                    <Text style={[styles.ruleDot, i === 0 && styles.ruleDotPrimary]}>·</Text>
                    <Text style={[styles.ruleText, i === 0 && styles.ruleTextPrimary]}>{r}</Text>
                  </View>
                ))}
              </View>
            </Pressable>
          );
        })}
      </View>

      {/* 战绩 */}
      {stats.rounds > 0 && (
        <View style={styles.statsRow}>
          <Stat label="局数" value={String(stats.rounds)} />
          <Stat label="赢" value={String(stats.won)} />
          <Stat label="胜率" value={`${Math.round(stats.winRate * 100)}%`} />
          <Stat label="最高分" value={String(stats.best)} />
        </View>
      )}

      <Text style={styles.footnote}>
        完全离线 · 无网络权限 · 模型 236 KB · 数据来自 Google QuickDraw（CC BY 4.0）
      </Text>
    </ScrollView>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Palette.paper, paddingHorizontal: Space.lg },

  brand: { gap: Space.sm, marginBottom: Space.xl },
  brandRow: { flexDirection: 'row', alignItems: 'center', gap: 9 },
  inkDot: { width: 13, height: 13, borderRadius: 7, backgroundColor: Palette.ink },
  brandName: {
    color: Palette.ink,
    fontSize: 27,
    fontWeight: '700',
    fontFamily: Typography.title,
    letterSpacing: -0.4,
  },
  brandLine: {
    color: Palette.inkSoft,
    fontSize: 13.5,
    lineHeight: 21,
    fontFamily: Typography.zh,
  },
  statusPill: {
    alignSelf: 'flex-start',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: Radius.pill,
    backgroundColor: Palette.paperSunken,
  },
  statusPillError: { backgroundColor: Palette.sealWash },
  statusDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: Palette.inkFaint },
  statusDotReady: { backgroundColor: Palette.jade },
  statusText: { color: Palette.inkSoft, fontSize: 11, fontFamily: Typography.mono },

  modes: { gap: Space.md },
  card: {
    backgroundColor: Palette.card,
    borderRadius: Radius.lg,
    borderWidth: 1,
    borderColor: Palette.rule,
    padding: Space.lg,
    gap: Space.sm,
  },
  cardPrimary: { backgroundColor: Palette.ink, borderColor: Palette.ink },
  cardPressed: { opacity: 0.88 },
  cardDisabled: { opacity: 0.45 },
  cardHead: { flexDirection: 'row', alignItems: 'center', gap: Space.md },
  glyphBox: {
    width: 38,
    height: 38,
    borderRadius: Radius.md,
    backgroundColor: Palette.paperSunken,
    alignItems: 'center',
    justifyContent: 'center',
  },
  glyphBoxPrimary: { backgroundColor: 'rgba(251,250,246,0.14)' },
  cardTitle: { color: Palette.ink, fontSize: 18, fontWeight: '700', fontFamily: Typography.zh },
  cardTitlePrimary: { color: Palette.paper },
  cardTagline: { color: Palette.inkSoft, fontSize: 12, fontFamily: Typography.zh, marginTop: 2 },
  cardTaglinePrimary: { color: 'rgba(251,250,246,0.62)' },

  ruleList: { gap: 3, paddingLeft: 2 },
  ruleRow: { flexDirection: 'row', gap: 7 },
  ruleDot: { color: Palette.inkFaint, fontSize: 12, lineHeight: 18 },
  ruleDotPrimary: { color: 'rgba(251,250,246,0.4)' },
  ruleText: {
    flex: 1,
    color: Palette.inkSoft,
    fontSize: 12,
    lineHeight: 18,
    fontFamily: Typography.zh,
  },
  ruleTextPrimary: { color: 'rgba(251,250,246,0.72)' },

  statsRow: {
    flexDirection: 'row',
    marginTop: Space.xl,
    paddingVertical: Space.md,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: Palette.rule,
  },
  stat: { flex: 1, alignItems: 'center', gap: 2 },
  statValue: { color: Palette.ink, fontSize: 17, fontWeight: '700', fontFamily: Typography.mono },
  statLabel: { color: Palette.inkFaint, fontSize: 10.5, fontFamily: Typography.zh },

  footnote: {
    marginTop: Space.lg,
    color: Palette.inkFaint,
    fontSize: 10.5,
    lineHeight: 16,
    fontFamily: Typography.zh,
    textAlign: 'center',
  },
});
