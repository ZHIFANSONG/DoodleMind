"""生成两个极短的音效（盖印章 / 猜错），直接写 16-bit PCM WAV。

为什么自己合成而不是找素材：这两个音只是「木头盖章」和「一声闷响」，
用两个正弦加指数包络就够，几百字节，不牵扯任何授权，也不会让仓库里多两个
来源不明的二进制文件。

用法：python tools/make_sounds.py
产物：assets/sounds/{stamp,nope}.wav
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "assets" / "sounds"
RATE = 22050


def env(n: int, tau: float) -> np.ndarray:
    """指数衰减包络，tau 是衰减到 1/e 的秒数"""
    t = np.arange(n) / RATE
    return np.exp(-t / tau)


def attack(x: np.ndarray, ms: float = 3.0) -> np.ndarray:
    """前几毫秒淡入，避免爆音（直接给正弦从 1 开始会有咔哒声）"""
    n = max(1, int(RATE * ms / 1000))
    head = np.linspace(0.0, 1.0, n)
    out = x.copy()
    out[:n] *= head
    return out


def write_wav(path: Path, samples: np.ndarray) -> None:
    peak = float(np.max(np.abs(samples))) or 1.0
    data = np.clip(samples / peak * 0.92, -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(pcm.tobytes())
    print(f"{path.relative_to(path.parent.parent.parent)}  {len(pcm) * 2} bytes")


def stamp() -> np.ndarray:
    """盖章：低频两音 + 一小段噪声当纸的摩擦声"""
    n = int(RATE * 0.22)
    t = np.arange(n) / RATE
    body = 0.9 * np.sin(2 * math.pi * 96 * t) + 0.5 * np.sin(2 * math.pi * 152 * t)
    body *= env(n, 0.045)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1, n) * env(n, 0.012)
    return attack(body + 0.35 * noise, 2.0)


def nope() -> np.ndarray:
    """猜错：一声往下滑的闷响，短、不刺耳"""
    n = int(RATE * 0.26)
    t = np.arange(n) / RATE
    # 220Hz 滑到 150Hz，滑动让人听出「泄气」而不是「提示」
    freq = 220 - 70 * (t / t[-1])
    phase = 2 * math.pi * np.cumsum(freq) / RATE
    body = 0.8 * np.sin(phase) + 0.25 * np.sin(2 * phase)
    body *= env(n, 0.06)
    return attack(body, 4.0)


if __name__ == "__main__":
    write_wav(OUT / "stamp.wav", stamp())
    write_wav(OUT / "nope.wav", nope())
