"""発話区間検出（VAD）。20ms フレームの音量を見て、ひと続きの発話を切り出す。

Whisper は「音声を少しずつ流し込む」使い方には向かず、ひとかたまりの音声を渡して
まとめて文字にする。そのため「話し始め〜話し終わり」を区切って渡す必要がある。
"""

from collections import deque

import numpy as np

FRAME_MS = 20


def rms(frame: bytes) -> float:
    """1 フレームの音量（二乗平均平方根）。無音で 0 付近、普通の声で数百〜数千。"""
    x = np.frombuffer(frame, dtype="<i2").astype(np.float32)
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


class Segmenter:
    """フレームを 1 つずつ feed() し、発話が終わったらその発話の PCM を返す。

    threshold     : この RMS 以上を「声あり」とみなす
    silence_ms    : 声なしがこれだけ続いたら発話の終わり
    min_speech_ms : これより短い発話は咳・物音として捨てる
    preroll_ms    : 声を検出する直前のこの長さを発話の頭に付け足す（子音の頭切れ対策）
    tail_ms       : 発話の後ろに残す無音。silence_ms 待った分のうち、これを超える無音は捨てる
    """

    def __init__(
        self,
        threshold: float = 500,
        silence_ms: int = 700,
        min_speech_ms: int = 250,
        preroll_ms: int = 200,
        tail_ms: int = 200,
    ) -> None:
        self.threshold = threshold
        self.silence_frames = silence_ms // FRAME_MS
        self.min_speech_frames = min_speech_ms // FRAME_MS
        self.tail_frames = tail_ms // FRAME_MS
        self.preroll: deque[bytes] = deque(maxlen=preroll_ms // FRAME_MS)

        self.in_speech = False  # いま発話中か
        self.buf = bytearray()  # 発話中に集めた PCM
        self.speech_run = 0  # 発話中の「声あり」フレーム数
        self.silence_run = 0  # 発話中に連続した「声なし」フレーム数

    def feed(self, frame: bytes) -> bytes | None:
        """フレームを 1 つ受け取る。発話が終わった瞬間だけ、その発話の PCM を返す。それ以外は None。"""
        voiced = rms(frame) >= self.threshold

        if not self.in_speech:
            if not voiced:
                self.preroll.append(frame)  # 古いものから自動で押し出される
                return None
            self.in_speech = True
            self.buf.extend(b"".join(self.preroll))
            self.preroll.clear()

        self.buf.extend(frame)
        if voiced:
            self.speech_run += 1
            self.silence_run = 0
        else:
            self.silence_run += 1
            if self.silence_run >= self.silence_frames:
                drop = (self.silence_run - self.tail_frames) * len(frame)
                pcm = bytes(self.buf[: len(self.buf) - drop]) if self.speech_run >= self.min_speech_frames else None
                self._reset()
                return pcm

        return None

    def reset(self) -> None:
        """途中まで集めた発話もプリロールも捨てて、まっさらな「発話前」に戻す。"""
        self._reset()
        self.preroll.clear()

    def _reset(self) -> None:
        self.in_speech = False
        self.buf.clear()
        self.speech_run = 0
        self.silence_run = 0
