"""音声認識（STT）。mlx-whisper で Mac 上だけで文字起こしする。

モデルは初回だけ Hugging Face から取得し ~/.cache/huggingface に置かれる。
以降はネットワークなしで動く。
"""

import mlx_whisper
import numpy as np

MODEL = "mlx-community/whisper-large-v3-turbo"
LANGUAGE = "ja"


def pcm8k_to_float16k(pcm: bytes) -> np.ndarray:
    """AudioSocket の音声（int16 / 8kHz）を Whisper の入力形式（float32 / 16kHz）に変換する。

    1. int16 を -1.0〜1.0 の float32 にする（32768 で割る）
    2. サンプルの間を直線でつないで 2 倍に増やし、8kHz → 16kHz にする
    電話の音声はもともと 4kHz までの成分しか持たないので、単純な補間でも認識精度は落ちない。
    """
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    src = np.arange(len(x))
    dst = np.arange(len(x) * 2) / 2
    return np.interp(dst, src, x).astype(np.float32)


def transcribe(pcm: bytes) -> str:
    """1 発話ぶんの PCM を文字にして返す。数百 ms〜数秒かかるブロッキング処理。"""
    result = mlx_whisper.transcribe(
        pcm8k_to_float16k(pcm),
        path_or_hf_repo=MODEL,
        language=LANGUAGE,
    )
    return result["text"].strip()


def warmup() -> None:
    """モデルを読み込んでおく。最初の通話で数秒待たされるのを防ぐため、起動時に 1 回呼ぶ。"""
    transcribe(bytes(8000))  # 4000 サンプル = 0.5 秒の無音
