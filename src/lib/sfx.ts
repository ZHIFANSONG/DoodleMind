/**
 * 两个音效：过关「盖章」、失败「闷响」。
 *
 * 三个约束决定了写法：
 * 1. **不能打断音乐**。`playsInSilentMode: false` 让 iOS 静音键对它生效——
 *    玩游戏时手机开着静音的人，不该被一个猜词小游戏突然出声。
 * 2. **播放器要复用一个**。每次 create 都会重新解码整个文件，而这两个音在
 *    一局里可能响很多次。
 * 3. **失败必须无声失败**。音效是锦上添花，任何平台的解码/权限问题都不该
 *    让结算卡弹不出来——所以每个调用点都包了 try/catch。
 */

import { createAudioPlayer, setAudioModeAsync, type AudioPlayer } from 'expo-audio';

export type Sfx = 'stamp' | 'nope';

const SOURCES: Record<Sfx, number> = {
  stamp: require('../../assets/sounds/stamp.wav'),
  nope: require('../../assets/sounds/nope.wav'),
};

const players: Partial<Record<Sfx, AudioPlayer>> = {};
let modeReady = false;

function ensureMode() {
  if (modeReady) return;
  modeReady = true;
  // 静音键生效；不打断别的 App 正在放的音频
  void setAudioModeAsync({ playsInSilentMode: false, interruptionMode: 'mixWithOthers' }).catch(() => {});
}

function playerFor(name: Sfx): AudioPlayer | null {
  try {
    const existing = players[name];
    if (existing) return existing;
    const p = createAudioPlayer(SOURCES[name]);
    p.volume = 0.85;
    players[name] = p;
    return p;
  } catch {
    return null;
  }
}

/** 播一次。重播要先回到 0，否则第二次调用会从上次结束的位置开始（等于没声音）。 */
export function playSfx(name: Sfx): void {
  ensureMode();
  const p = playerFor(name);
  if (!p) return;
  try {
    // seekTo 是异步的，但在两个平台上都是「立刻把播放头挪回去」，
    // 紧跟着的 play() 不会等它——这里不 await，是为了不让音效拖慢结算动画。
    void p.seekTo(0).catch(() => {});
    p.play();
  } catch {
    /* 静默失败 */
  }
}
