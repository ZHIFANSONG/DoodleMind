/**
 * 画册：所有画过的局，按时间倒序。
 *
 * 缩略图是**现画的**——从存下来的矢量笔画重绘，不是截图。
 * 好处：任意尺寸都清晰，一条记录只占几百字节，
 * 而且以后改配色/改笔触，历史缩略图会一起变好看。
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { SymbolView } from 'expo-symbols';
import Svg, { Path } from 'react-native-svg';

import { toPathD, type Pt } from '@/components/drawing-canvas';
import { Palette, Radius, Space, Typography } from '@/constants/design';
import { clearHistory, loadHistory, removeRecord, summarize, unpackStrokes, type GameRecord } from '@/lib/history';

export default function GalleryScreen() {
  const insets = useSafeAreaInsets();
  const { width } = useWindowDimensions();
  const [list, setList] = useState<GameRecord[]>([]);
  const [loading, setLoading] = useState(true);

  const reload = useCallback(() => {
    void loadHistory().then((v) => {
      setList(v);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  const stats = useMemo(() => summarize(list), [list]);

  const onDelete = useCallback(
    (id: string) => {
      void removeRecord(id).then(setList);
    },
    [],
  );

  const onClearAll = useCallback(() => {
    Alert.alert('清空画册', '所有记录都会被删掉，包括那些你画得很得意的。', [
      { text: '算了', style: 'cancel' },
      {
        text: '清空',
        style: 'destructive',
        onPress: () => {
          void clearHistory().then(() => setList([]));
        },
      },
    ]);
  }, []);

  // 两列瀑布流不必要：记录是等高的正方形缩略图，两列刚好
  const columnWidth = (width - Space.lg * 2 - Space.md) / 2;

  return (
    <View style={[styles.root, { paddingTop: insets.top + Space.sm }]}>
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>画册</Text>
          <Text style={styles.subtitle}>
            {stats.rounds} 局 · 猜中 {stats.cleared} · 最高 {stats.best} 分
          </Text>
        </View>
        <View style={{ flex: 1 }} />
        {list.length > 0 && (
          <Pressable onPress={onClearAll} hitSlop={10} style={styles.clearBtn}>
            <Text style={styles.clearBtnText}>清空</Text>
          </Pressable>
        )}
      </View>

      {stats.totalStrokes > 0 && (
        <Text style={styles.metaLine}>
          共 {stats.totalStrokes} 笔 · 平均用时 {(stats.avgMs / 1000).toFixed(1)}s
        </Text>
      )}

      <FlatList
        data={list}
        keyExtractor={(r) => r.id}
        numColumns={2}
        columnWrapperStyle={{ gap: Space.md }}
        contentContainerStyle={{
          gap: Space.md,
          paddingTop: Space.md,
          paddingBottom: insets.bottom + 80,
        }}
        ListEmptyComponent={
          <View style={styles.empty}>
            <SymbolView
              name="rectangle.and.pencil.and.ellipsis"
              tintColor={Palette.inkFaint}
              size={44}
              resizeMode="scaleAspectFit"
            />
            <Text style={styles.emptyText}>
              {loading ? '读取中…' : '还没有画。去画几张，AI 会记住它猜你说的每一句。'}
            </Text>
          </View>
        }
        renderItem={({ item }) => (
          <RecordCard record={item} size={columnWidth} onDelete={() => onDelete(item.id)} />
        )}
      />
    </View>
  );
}

function RecordCard({
  record,
  size,
  onDelete,
}: {
  record: GameRecord;
  size: number;
  onDelete: () => void;
}) {
  const strokes = useMemo(() => unpackStrokes(record.strokes), [record.strokes]);

  return (
    <Pressable
      onLongPress={onDelete}
      delayLongPress={450}
      style={[styles.card, { width: size }]}>
      <View style={[styles.thumb, { width: size, height: size }]}>
        <Svg width={size} height={size}>
          {strokes.map((s, i) => (
            <Path
              key={i}
              d={toPathD(s.pts as Pt[], size)}
              stroke={s.color}
              strokeWidth={Math.max(1.4, s.width * size)}
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}
        </Svg>
        <View style={[styles.badge, record.cleared ? styles.badgeOk : styles.badgeMiss]}>
          <Text style={[styles.badgeText, record.cleared ? styles.badgeTextOk : styles.badgeTextMiss]}>
            {record.cleared ? `+${record.score}` : '未中'}
          </Text>
        </View>
      </View>
      <View style={styles.cardFoot}>
        <Text style={styles.cardWord} numberOfLines={1}>
          {record.emoji} {record.targetZh}
        </Text>
        <Text style={styles.cardMeta}>
          {(record.elapsedMs / 1000).toFixed(1)}s · {record.strokeCount} 笔
        </Text>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Palette.paper, paddingHorizontal: Space.lg },
  header: { flexDirection: 'row', alignItems: 'flex-end' },
  title: { color: Palette.ink, fontSize: 30, fontWeight: '700', fontFamily: Typography.zh },
  subtitle: { color: Palette.inkFaint, fontSize: 12, fontFamily: Typography.zh, marginTop: 2 },
  metaLine: { color: Palette.inkFaint, fontSize: 11, fontFamily: Typography.mono, marginTop: 4 },
  clearBtn: {
    paddingHorizontal: Space.md,
    paddingVertical: 6,
    borderRadius: Radius.pill,
    borderWidth: 1,
    borderColor: Palette.rule,
  },
  clearBtnText: { color: Palette.inkSoft, fontSize: 12, fontFamily: Typography.zh },

  card: {
    backgroundColor: Palette.card,
    borderRadius: Radius.md,
    borderWidth: 1,
    borderColor: Palette.rule,
    overflow: 'hidden',
  },
  thumb: { backgroundColor: Palette.card },
  badge: {
    position: 'absolute',
    right: 6,
    top: 6,
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: Radius.pill,
  },
  badgeOk: { backgroundColor: Palette.jadeWash },
  badgeMiss: { backgroundColor: Palette.paperSunken },
  badgeText: { fontSize: 10, fontFamily: Typography.mono, fontWeight: '700' },
  badgeTextOk: { color: Palette.jade },
  badgeTextMiss: { color: Palette.inkFaint },
  cardFoot: { paddingHorizontal: Space.sm, paddingVertical: 7, gap: 1 },
  cardWord: { color: Palette.ink, fontSize: 13, fontFamily: Typography.zh, fontWeight: '600' },
  cardMeta: { color: Palette.inkFaint, fontSize: 10, fontFamily: Typography.mono },

  empty: { alignItems: 'center', gap: Space.md, paddingTop: 90, paddingHorizontal: Space.xl },
  emptyText: { color: Palette.inkFaint, fontSize: 13, fontFamily: Typography.zh, textAlign: 'center', lineHeight: 20 },
});


