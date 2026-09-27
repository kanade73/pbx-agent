"""音声合成（TTS）。macOS 標準の say コマンドで、電話にそのまま流せる PCM を作る。

say は --data-format=LEI16@8000 を指定すると、AudioSocket と同じ
signed linear 16bit / 8kHz / mono で書き出せるので、変換（リサンプル）が要らない。
"""

import subprocess
import tempfile
import wave

VOICE = "Kyoko"


def synthesize(text: str) -> bytes:
    """テキストを読み上げた PCM（int16 / 8kHz / mono）を返す。数百 ms かかるブロッキング処理。

    say は標準出力に音声を出せないので、一時ファイルに WAV で書かせてから読み戻す。
    WAV のヘッダは wave モジュールが読み飛ばし、中身の PCM だけを取り出す。
    """
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        subprocess.run(
            ["say", "-v", VOICE, "--data-format=LEI16@8000", "--file-format=WAVE", "-o", f.name, text],
            check=True,
        )
        with wave.open(f.name, "rb") as w:
            return w.readframes(w.getnframes())
