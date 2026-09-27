"""音声認識（STT）。バックエンドはマシンに合わせて自動で選び、環境変数で上書きもできる。

    STT_BACKEND=auto    既定。Apple Silicon なら mlx、それ以外は faster（_detect() 参照）
    STT_BACKEND=mlx     mlx-whisper（Apple Silicon の GPU で動く）
    STT_BACKEND=faster  faster-whisper（CTranslate2。NVIDIA GPU があれば GPU、なければ CPU。
                        `uv sync --extra faster` でインストールしておく）
    STT_MODEL=...       モデル名を上書きしたいときだけ指定する

モデルは初回だけ Hugging Face から取得し ~/.cache/huggingface に置かれる。
以降はネットワークなしで動く（HF_HUB_OFFLINE=1 で起動すれば取得を試みない）。
"""

import importlib.util
import os
import platform
from typing import Protocol

import numpy as np

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


class Backend(Protocol):
    """STT バックエンドが満たすべき形。name はログ表示用。"""

    name: str

    def transcribe(self, audio: np.ndarray) -> str:
        """float32 / 16kHz の音声を受け取り、認識したテキストを返す。"""
        ...


class MlxWhisper:
    """mlx-whisper。Apple Silicon の GPU（Metal）で動くので Mac では最速。"""

    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo") -> None:
        import mlx_whisper  # Mac 以外では入っていないので、使うときだけ import する

        self._mlx = mlx_whisper
        self.model = model
        self.name = f"mlx-whisper {model}"

    def transcribe(self, audio: np.ndarray) -> str:
        result = self._mlx.transcribe(audio, path_or_hf_repo=self.model, language=LANGUAGE)
        return result["text"].strip()


class FasterWhisper:
    """faster-whisper（CTranslate2）。NVIDIA GPU があれば GPU、なければ CPU で動く。

    Apple の GPU には対応していないので、Mac では CPU 実行になる（mlx より遅い）。
    """

    def __init__(self, model: str | None = None) -> None:
        import ctranslate2
        from faster_whisper import WhisperModel

        if ctranslate2.get_cuda_device_count() > 0:
            device, compute_type, default_model = "cuda", "float16", "large-v3-turbo"
        else:
            # CPU では int8 に量子化して速度を稼ぐ。large は CPU だと数秒かかるので small を既定に
            device, compute_type, default_model = "cpu", "int8", "small"
        model = model or default_model
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.name = f"faster-whisper {model} ({device}/{compute_type})"

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return "".join(segment.text for segment in segments).strip()


def _detect() -> str:
    """このマシンで一番速いバックエンドを選ぶ。

    Apple Silicon の Mac で mlx-whisper が入っていれば mlx（GPU を使える）。
    それ以外（Linux / Intel Mac / Windows）は faster。faster の中で CUDA か CPU かをさらに判定する。
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if importlib.util.find_spec("mlx_whisper") is not None:
            return "mlx"
    return "faster"


def _create() -> Backend:
    backend = os.environ.get("STT_BACKEND") or "auto"  # 空文字（STT_BACKEND=）も auto 扱い
    if backend == "auto":
        backend = _detect()
    model = os.environ.get("STT_MODEL") or None
    if backend == "mlx":
        return MlxWhisper(model) if model else MlxWhisper()
    if backend == "faster":
        return FasterWhisper(model)
    raise ValueError(f"STT_BACKEND は auto / mlx / faster のどれか: {backend!r}")


_backend: Backend | None = None


def backend() -> Backend:
    """使用中のバックエンド。初回呼び出し時に STT_BACKEND を見て作る。"""
    global _backend
    if _backend is None:
        _backend = _create()
    return _backend


def transcribe(pcm: bytes) -> str:
    """1 発話ぶんの PCM（int16 / 8kHz）を文字にして返す。数百 ms〜数秒かかるブロッキング処理。"""
    return backend().transcribe(pcm8k_to_float16k(pcm))


def warmup() -> None:
    """モデルを読み込んでおく。最初の通話で数秒待たされるのを防ぐため、起動時に 1 回呼ぶ。"""
    transcribe(bytes(8000))  # 4000 サンプル = 0.5 秒の無音
