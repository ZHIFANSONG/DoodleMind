/**
 * 画册：每一局都留一笔。
 *
 * 缩略图是**当场把矢量笔画重画出来的**，不是截图。好处：
 * 1. 存储省 20 倍（8~20KB vs 400KB/张）
 * 2. 换设备、换分辨率永远清晰
 * 3. 可以在缩略图上做交互——比如点开看「模型当时看到了什么」
 *    如果存的是 PNG，这一层就永远失去了。
 *
 * 代价是渲染成本：一屏 12 张卡片、每张 3~8 条 Path，滚动时会有明显开销。
 * 所以 FlatList 开了 removeClippedSubviews + 固定高度 getItemLayout，
 * 并且缩略图 Svg 只画路径不画压感分档（那条路径每张只要 1 个 Path）。
 */

import { useCallback, useState } from 'react';
import {
  Alert,
  FlatList,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useScreenTop } from '@/hooks/use-screen-top';
import { useFocusEffect } from 'expo-router';
import * as Haptics from 'expo-haptics';
import Svg, { Path } from 'react-native-svg';
import { MenuView } from '@expo/ui/community/menu';

import { Glyph } from '@/components/glyph';
import { EmptyState } from '@/components/native/kit';

import { toPathD } from '@/components/drawing-canvas';
import { Palette, Radius, Space, Typography } from '@/constants/design';
import {
  clearHistory,
  loadHistory,
  removeRecord,
  summarize,
  unpackStrokes,
  type GameRecord,
} from '@/lib/history';
import { MODES } from '@/lib/modes';

/**
 * MenuView 只有 iOS / Android 有原生实现。
 * web 版（`node_modules/@expo/ui/build/community/menu/MenuView.js`）只渲染 trigger、
 * 不弹菜单，还会打一次 console.warn——所以在 web 上退回原生的长按删除。
 */
const NATIVE_MENU = Platform.OS === 'ios' || Platform.OS === 'android';

export default function GalleryScreen() {
  const insets = useSafeAreaInsets();
  const padTop = useScreenTop();
  const { width } = useWindowDimensions();
  const [records, setRecords] = useState<GameRecord[]>([]);
  const [loaded, setLoaded] = useState(false);

  // 每次切回本页都重新读：游戏页写盘是异步的，用全局状态同步反而更复杂
  useFocusEffect(
    useCallback(() => {
      let alive = true;
      void loadHistory().then((list) => {
        if (!alive) return;
        setRecords(list);
        setLoaded(true);
      });
      return () => {
        alive = false;
      };
    }, []),
  );

  const stats = summarize(records);
  const cardW = (width - Space.lg * 2 - Space.md) / 2;

  const onDelete = useCallback((id: string, word: string) => {
    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    Alert.alert('删掉这一局？', `${word} · 删了就找不回来了`, [
      { text: '算了', style: 'cancel' },
      {
        text: '删',
        style: 'destructive',
        onPress: () => {
          void removeRecord(id).then(setRecords);
        },
      },
    ]);
  }, []);

  const onClearAll = useCallback(() => {
    Alert.alert('清空全部记录？', `共 ${records.length} 局，会一并删掉`, [
      { text: '算了', style: 'cancel' },
      {
        text: '清空',
        style: 'destructive',
        onPress: () => {
          void clearHistory().then(() => setRecords([]));
        },
      },
    ]);
  }, [records.length]);

  if (loaded && records.length === 0) {
    return (
      <View style={[styles.empty, { paddingTop: padTop + Space.xxl }]}>
        {/*
          iOS 上是系统 `ContentUnavailableView`——和「设置」里没开过某个功能、
          App Store 里搜不到东西时同一个控件：图标、标题、说明的间距与字重
          都由 UIKit 排，跟随 Dynamic Type，VoiceOver 也读得对。
          web/Android 退回我们自己摆的三行。
        */}
        {Platform.OS === 'ios' ? (
          <EmptyState title="还什么都没画" description="回去画一局，这里会留下你的每一笔" />
        ) : (
          <>
            <Glyph name="emptyGallery" color={Palette.inkFaint} size={46} />
            <Text style={styles.emptyTitle}>还什么都没画</Text>
            <Text style={styles.emptyText}>回去画一局，这里会留下你的每一笔</Text>
          </>
        )}
      </View>
    );
  }

  return (
    <FlatList
      style={styles.root}
      contentContainerStyle={{
        paddingTop: padTop + Space.md,
        paddingBottom: insets.bottom + Space.xxl,
        paddingHorizontal: Space.lg,
      }}
      data={records}
      keyExtractor={(r) => r.id}
      numColumns={2}
      columnWrapperStyle={{ gap: Space.md }}
      ItemSeparatorComponent={() => <View style={{ height: Space.md }} />}
      removeClippedSubviews
      initialNumToRender={8}
      maxToRenderPerBatch={8}
      ListHeaderComponent={
        <View style={styles.header}>
          <View style={styles.statRow}>
            <Stat label="局数" value={String(stats.rounds)} />
            <Stat label="赢" value={String(stats.won)} />
            <Stat label="胜率" value={`${Math.round(stats.winRate * 100)}%`} />
            <Stat label="最高分" value={String(stats.best)} />
          </View>
          <Text style={styles.subMeta}>
            共 {stats.totalStrokes.toLocaleString()} 笔 · 累计 {fmtDuration(stats.totalMs)}
          </Text>
          <Pressable onPress={onClearAll} hitSlop={8} style={styles.clearBtn}>
            <Text style={styles.clearBtnText}>清空全部</Text>
          </Pressable>
        </View>
      }
      renderItem={({ item }) => (
        <RecordCard item={item} width={cardW} onDelete={() => onDelete(item.id, item.targetZh)} />
      )}
    />
  );
}

function fmtDuration(ms: number): string {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s} 秒`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} 分 ${s % 60} 秒`;
  return `${Math.floor(m / 60)} 时 ${m % 60} 分`;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

const RecordCard = ({
  item,
  width,
  onDelete,
}: {
  item: GameRecord;
  width: number;
  onDelete: () => void;
}) => {
  const strokes = unpackStrokes(item.strokes);
  const size = width - 20;
  const mode = MODES[item.mode] ?? MODES.challenge;

  const outcomeColor =
    item.outcome === 'win' ? Palette.jade : item.outcome === 'lose' ? Palette.seal : Palette.inkFaint;
  const outcomeText = item.outcome === 'win' ? '赢' : item.outcome === 'lose' ? '输' : '平';

  const card = (
    <View style={[styles.card, { width }]}>
      <View style={[styles.thumb, { width: size, height: size }]}>
        <Svg width={size} height={size}>
          {strokes.map((s, i) => (
            <Path
              key={i}
              d={toPathD(s.pts, size)}
              stroke={s.color}
              strokeWidth={Math.max(1.4, s.width * size)}
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}
        </Svg>
      </View>
      <View style={styles.cardFoot}>
        <View style={{ flex: 1 }}>
          <Text style={styles.cardWord} numberOfLines={1}>
            {item.emoji} {item.targetZh}
          </Text>
          <Text style={styles.cardMeta} numberOfLines={1}>
            {mode.title} · {item.strokeCount} 笔 · {Math.round(item.elapsedMs / 1000)}s
          </Text>
        </View>
        <View style={[styles.badge, { backgroundColor: outcomeColor }]}>
          <Text style={styles.badgeText}>{item.score > 0 ? `+${item.score}` : outcomeText}</Text>
        </View>
      </View>
    </View>
  );

  if (NATIVE_MENU) {
    // iOS/Android：长按弹系统自己的上下文菜单。整个卡片会被系统挖成圆角、
    // 背景压暗、其余元素轻微缩放——这套动画由 UIKit/Compose 播放，JS 复刻不出来。
    return (
      <MenuView
        shouldOpenOnLongPress
        actions={[{ id: 'delete', title: '删掉这一局', image: 'trash', attributes: { destructive: true } }]}
        onPressAction={(e) => {
          if (e.nativeEvent.event === 'delete') onDelete();
        }}
        style={{ width }}>
        {card}
      </MenuView>
    );
  }

  // web：MenuView 只渲染 trigger、不弹菜单（会打一次 console.warn），
  // 所以退回长按直接走删除确认。
  return (
    <Pressable onLongPress={onDelete} delayLongPress={450}>
      {card}
    </Pressable>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Palette.paper },

  header: { paddingBottom: Space.lg },
  title: { color: Palette.ink, fontSize: 24, fontWeight: '700', fontFamily: Typography.title },
  statRow: {
    flexDirection: 'row',
    marginTop: Space.md,
    paddingVertical: Space.md,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: Palette.rule,
  },
  stat: { flex: 1, alignItems: 'center', gap: 2 },
  statValue: { color: Palette.ink, fontSize: 16, fontWeight: '700', fontFamily: Typography.mono },
  statLabel: { color: Palette.inkFaint, fontSize: 10, fontFamily: Typography.zh },
  subMeta: { color: Palette.inkFaint, fontSize: 11, fontFamily: Typography.zh, marginTop: Space.sm },
  clearBtn: { alignSelf: 'flex-start', marginTop: Space.sm },
  clearBtnText: { color: Palette.seal, fontSize: 11.5, fontFamily: Typography.zh },

  card: {
    backgroundColor: Palette.card,
    borderRadius: Radius.md,
    borderWidth: 1,
    borderColor: Palette.rule,
    padding: 9,
    gap: 7,
  },
  thumb: { backgroundColor: Palette.paper, borderRadius: Radius.sm, overflow: 'hidden' },
  cardFoot: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  cardWord: { color: Palette.ink, fontSize: 12.5, fontFamily: Typography.zh, fontWeight: '600' },
  cardMeta: { color: Palette.inkFaint, fontSize: 9.5, fontFamily: Typography.zh, marginTop: 1 },
  badge: { paddingHorizontal: 6, paddingVertical: 2, borderRadius: Radius.sm },
  badgeText: { color: '#FFF', fontSize: 9.5, fontWeight: '700', fontFamily: Typography.mono },

  empty: { flex: 1, alignItems: 'center', gap: Space.sm, backgroundColor: Palette.paper },
  emptyTitle: { color: Palette.inkSoft, fontSize: 16, fontFamily: Typography.zh, fontWeight: '600' },
  emptyText: { color: Palette.inkFaint, fontSize: 12.5, fontFamily: Typography.zh },
});
