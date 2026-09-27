"""応答生成（LLM）。Mac 上の Ollama（http://localhost:11434）に問い合わせる。

外部ライブラリを増やさないよう、標準ライブラリの urllib だけで HTTP を話す。
"""

import json
import time
import urllib.request
from collections.abc import Iterator

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"
KEEP_ALIVE = "30m"  # この間リクエストがなくてもモデルをメモリに載せたままにする
MAX_HISTORY_MESSAGES = 10  # 直近 5 往復ぶん

SYSTEM_PROMPT = (
    "あなたは電話の自動応答アシスタントです。"
    "相手の発言は音声認識の結果なので、誤字や聞き間違いを含むことがあります。"
    "返答は電話で読み上げるので、1〜2文の短い話し言葉の日本語で答えてください。"
    "箇条書き・記号・絵文字は使わないでください。"
    "知らないこと（予定・個人情報など）は推測で答えず、分からないと伝えてください。"
)


def stream_chat(messages: list[dict]) -> Iterator[str]:
    """Ollama にメッセージ列を送り、返答を届いた順に少しずつ返す（ブロッキング）。

    Ollama はストリーミング時に 1 行 1 JSON（NDJSON）で返してくるので、
    行ごとに読んで message.content を取り出す。
    """
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "think": False,  # qwen3 の思考モードを切る（切らないと答える前に数秒考え込む）
        "keep_alive": KEEP_ALIVE,
    }
    req = urllib.request.Request(
        OLLAMA_URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            piece = json.loads(line).get("message", {}).get("content", "")
            if piece:
                yield piece


SENTENCE_ENDS = "。！？!?\n"


def stream_sentences(messages: list[dict]) -> Iterator[str]:
    """stream_chat() の断片をつなぎ、文が完成するたびに 1 文ずつ返す（ブロッキング）。

    全文を待たずに最初の 1 文から読み上げを始めるための区切り。
    """
    buf = ""
    for piece in stream_chat(messages):
        buf += piece
        while True:
            cut = next((i for i, ch in enumerate(buf) if ch in SENTENCE_ENDS), -1)
            if cut < 0:
                break
            sentence, buf = buf[: cut + 1].strip(), buf[cut + 1 :]
            if sentence:
                yield sentence
    if buf.strip():
        yield buf.strip()


def reply(messages: list[dict]) -> tuple[str, float]:
    """返答を最後まで受け取り、(全文, 最初の文字が届くまでの秒数) を返す。

    いまはログに出すだけなので全文を待つ。TTS を付けるときは stream_chat() を直接使い、
    文の区切りごとに読み上げを始めることで応答の出だしを早くする。
    """
    started = time.monotonic()
    first = None
    parts = []
    for piece in stream_chat(messages):
        if first is None:
            first = time.monotonic() - started
        parts.append(piece)
    return "".join(parts).strip(), first or 0.0


def warmup() -> None:
    """モデルをメモリに読み込ませる。初回は 10 秒以上かかるので、起動時に済ませておく。"""
    reply([{"role": "user", "content": "こんにちは"}])


class Conversation:
    """1 通話ぶんの会話履歴。通話ごとに 1 つ作る。"""

    def __init__(self) -> None:
        self.history: list[dict] = []  # system を除いた user / assistant の発言

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def messages(self) -> list[dict]:
        """LLM に送るメッセージ列を作る。先頭には必ず system プロンプトを置く。"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history[-MAX_HISTORY_MESSAGES:],
        ]

