"""AudioSocket サーバ（第1段階: オウム返し）

Asterisk の AudioSocket() から TCP で接続を受け、届いた音声をそのまま送り返す。
1 接続 = 1 通話。

AudioSocket のフレーム形式:
    [種類 1byte][長さ 2byte ビッグエンディアン][ペイロード 長さ分]

種類:
    0x00 HANGUP  通話終了（ペイロードなし）
    0x01 UUID    通話の識別子（16byte）。接続直後に 1 回だけ届く
    0x10 AUDIO   音声。signed linear 16bit / 8kHz / mono。通常 20ms = 320byte
    0xff ERROR   Asterisk 側のエラー
"""

import asyncio
import logging
import time
import uuid

from . import llm, stt, tts
from .vad import Segmenter

HOST = "0.0.0.0"
PORT = 9090
ECHO = False  # True にすると第1段階のオウム返しも続ける

KIND_HANGUP = 0x00
KIND_UUID = 0x01
KIND_AUDIO = 0x10
KIND_ERROR = 0xFF

log = logging.getLogger("voice-agent")

# Whisper は同時に 1 つずつ動かす（GPU/メモリを取り合わないように）
stt_lock = asyncio.Lock()


FRAME_BYTES = 320  # 20ms 分の PCM（8000Hz × 2byte × 0.02s）
FRAME_SEC = 0.02
# 読み上げが終わってからも、この秒数は「まだ話している」とみなす。
# 送った音声が Zoiper で鳴り、スピーカー → マイク → Asterisk → Python と戻ってくるまでの
# 遅れと、部屋の残響の分。短すぎると AI の語尾を自分の発言として拾ってしまう。
ECHO_TAIL_SEC = 0.5


class Player:
    """通話ごとの再生係。enqueue() された PCM を 20ms ずつ、実時間と同じ速さで Asterisk に送る。

    なぜ一定間隔で送るのか:
    合成した数秒分の音声を一度に write() すると、Asterisk は受け取った順に次々と
    チャネルへ流すため、早送りのように再生されたり、バッファから溢れた分が捨てられたりする。
    電話の音声は「20ms ごとに 1 フレーム」が前提なので、送る側がそのリズムを作る必要がある。
    """

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.playing = False
        self.last_played_at = 0.0  # 最後のフレームを送った時刻（loop.time()）
        self.task = asyncio.create_task(self._run())

    def enqueue(self, pcm: bytes) -> None:
        self.queue.put_nowait(pcm)

    def close(self) -> None:
        self.task.cancel()

    def is_speaking(self) -> bool:
        """AI がいま話している（か、話し終えて間もない）なら True。半二重の判定に使う。

        再生中・再生待ちがある・最後の音を送ってから ECHO_TAIL_SEC 以内、のどれかなら True。
        """
        if self.playing or not self.queue.empty():
            return True
        return asyncio.get_running_loop().time() - self.last_played_at < ECHO_TAIL_SEC

    async def _run(self) -> None:
        while True:
            pcm = await self.queue.get()
            self.playing = True
            try:
                await self._play(pcm)
            finally:
                self.playing = False
                self.last_played_at = asyncio.get_running_loop().time()

    async def _play(self, pcm: bytes) -> None:
        """pcm を FRAME_BYTES ずつに切り、FRAME_SEC 間隔で writer に送る。"""
        loop = asyncio.get_running_loop()
        next_frame_at = loop.time()
        for offset in range(0, len(pcm), FRAME_BYTES):
            await asyncio.sleep(max(0, next_frame_at - loop.time()))   # ① 予定時刻まで待つ
            self.writer.write(build_frame(KIND_AUDIO, pcm[offset : offset + FRAME_BYTES]))  # ② 送る
            await self.writer.drain()
            next_frame_at += FRAME_SEC                                  # ③ 次の予定 = 今の予定 + 20ms
            if loop.time() - next_frame_at > 0.1:                       # ④ 大きく遅れたときだけ
                next_frame_at = loop.time()                             #    予定を今に合わせ直す


async def on_utterance(
    call_id: uuid.UUID | None,
    pcm: bytes,
    conv: llm.Conversation,
    turn_lock: asyncio.Lock,
    player: Player,
) -> None:
    """1 発話ぶんの処理: 文字起こし → LLM → 1 文ずつ音声合成 → 再生キューへ。

    stt / llm / tts はどれも数百 ms〜数秒かかるブロッキング処理なので、
    asyncio.to_thread() で別スレッドに逃がす。そうしないと、その間 handle_call() の
    ループが止まり、Asterisk からの音声フレームを読めなくなる。

    turn_lock は通話ごとのロック。続けて話したとき、発話の順番どおりに履歴へ積むため。
    """
    seconds = len(pcm) / 2 / 8000
    async with turn_lock:
        async with stt_lock:
            started = time.monotonic()
            text = await asyncio.to_thread(stt.transcribe, pcm)
            elapsed = time.monotonic() - started
        log.info("認識 call=%s 発話=%.1f秒 処理=%.2f秒 「%s」", call_id, seconds, elapsed, text)
        if not text:
            return

        conv.add("user", text)
        started = time.monotonic()
        sentences = await start_sentence_stream(conv.messages())
        answer = []
        while (sentence := await sentences.get()) is not None:
            audio = await asyncio.to_thread(tts.synthesize, sentence)
            if not answer:
                log.info("最初の音声 call=%s %.2f秒後", call_id, time.monotonic() - started)
            player.enqueue(audio)
            answer.append(sentence)
        conv.add("assistant", "".join(answer))
        log.info("応答 call=%s 全体=%.2f秒 「%s」", call_id, time.monotonic() - started, "".join(answer))


async def start_sentence_stream(messages: list[dict]) -> asyncio.Queue[str | None]:
    """llm.stream_sentences() を別スレッドで回し、文ができるたびに asyncio.Queue に入れる。

    ブロッキングなジェネレータ（別スレッド）と asyncio（メインスレッド）の橋渡し。
    別スレッドから asyncio.Queue を直接触ると壊れるので、call_soon_threadsafe() で
    「メインスレッドで put してね」と依頼する。最後に None を入れて終わりを知らせる。
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def worker() -> None:
        try:
            for sentence in llm.stream_sentences(messages):
                loop.call_soon_threadsafe(queue.put_nowait, sentence)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, worker)
    return queue


class ArrivalStats:
    """AUDIO フレームの到着間隔（秒）を集めて、1 秒ごとに要約をログに出す。

    理想はきっちり 0.020 秒間隔。まとめて届いていれば、
    間隔 0 に近いものと、大きく空いたもの（例 0.1 秒）が混ざる。
    """

    REPORT_EVERY = 50  # 50 フレーム ≒ 1 秒ごとに要約

    def __init__(self) -> None:
        self.gaps: list[float] = []

    def add(self, gap: float) -> None:
        self.gaps.append(gap)
        if len(self.gaps) >= self.REPORT_EVERY:
            self.report()

    def report(self) -> None:
        if not self.gaps:
            return
        ms = [g * 1000 for g in self.gaps]
        bunched = sum(1 for m in ms if m < 5)  # ほぼ同時に届いた = まとめ届き
        late = sum(1 for m in ms if m > 30)  # 大きく遅れて届いた
        log.info(
            "到着間隔 n=%d 平均=%.1fms 最小=%.1fms 最大=%.1fms  まとめ届き(<5ms)=%d 遅れ(>30ms)=%d",
            len(ms), sum(ms) / len(ms), min(ms), max(ms), bunched, late,
        )
        self.gaps.clear()


async def read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """TCP から AudioSocket のフレームを 1 つ読んで (種類, ペイロード) を返す。

    まずヘッダ 3byte を読み、そこに書かれた長さの分だけペイロードを読む。
    TCP はデータの区切りを保証しない（320byte が 100+220 に分かれて届くこともある）ので、
    readexactly() で「指定バイト数そろうまで待つ」ことで 1 フレームずつ取り出している。

    Asterisk が接続を切ると asyncio.IncompleteReadError が送出される。
    """
    header = await reader.readexactly(3)
    kind = header[0]
    length = int.from_bytes(header[1:3], "big")
    payload = await reader.readexactly(length) if length else b""
    return kind, payload


def build_frame(kind: int, payload: bytes = b"") -> bytes:
    """read_frame() の逆。(種類, ペイロード) を送信用のバイト列に組み立てる。

    例: build_frame(KIND_AUDIO, pcm) → Asterisk に送ると、その音声が電話口で再生される。
        build_frame(KIND_HANGUP)     → Asterisk に通話を切らせる。
    """
    return bytes([kind]) + len(payload).to_bytes(2, "big") + payload


async def handle_call(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """1 通話ぶんの処理。Asterisk から接続が来るたびに asyncio が呼び出す。

    reader は Asterisk → Python（電話の相手の声が流れてくる）、
    writer は Python → Asterisk（ここに書いた音声が電話口で鳴る）。

    流れ:
        1. 接続直後に UUID フレームが 1 つ届く（どの通話かの識別子）
        2. 以降 20ms ごとに AUDIO フレームが届き続ける
        3. 通話が終わると HANGUP が届くか、接続そのものが切れる
    複数の通話が同時に来ても、それぞれ別の handle_call() として並行に動く。
    """
    peer = writer.get_extra_info("peername")
    log.info("接続 %s", peer)
    call_id = None
    audio_frames = 0
    stats = ArrivalStats()
    last_arrival: float | None = None
    segmenter = Segmenter()
    conv = llm.Conversation()
    turn_lock = asyncio.Lock()
    player = Player(writer)
    pending: set[asyncio.Task] = set()  # この通話で処理中の on_utterance

    try:
        while True:
            kind, payload = await read_frame(reader)

            if kind == KIND_UUID:
                call_id = uuid.UUID(bytes=payload)
                log.info("通話開始 call=%s", call_id)
            elif kind == KIND_AUDIO:
                audio_frames += 1
                now = time.monotonic()
                if last_arrival is not None:
                    stats.add(now - last_arrival)
                last_arrival = now

                if player.is_speaking():
                    segmenter.reset()
                    utterance = None
                else:
                    utterance = segmenter.feed(payload)
                if utterance:
                    task = asyncio.create_task(on_utterance(call_id, utterance, conv, turn_lock, player))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                if ECHO:
                    writer.write(build_frame(KIND_AUDIO, payload))
            elif kind == KIND_HANGUP:
                break
            elif kind == KIND_ERROR:
                log.warning("Asterisk 側のエラー call=%s", call_id)
                break
            else:
                log.warning("不明なフレーム種類 %02x call=%s", kind, call_id)

    except asyncio.IncompleteReadError:
        log.info("Asterisk 側が切断")
    finally:
        # 切断後に届く認識・応答は誰も聞かないので止める。
        # to_thread で走っている STT/LLM 自体は止められないが、その先の TTS・履歴追加は行われない
        for task in pending:
            task.cancel()
        if pending:
            log.info("切断のため処理中の発話 %d 件を取り消し call=%s", len(pending), call_id)
        player.close()
        stats.report()
        log.info("終了 call=%s audio_frames=%d (約 %.1f 秒)", call_id, audio_frames, audio_frames * 0.02)
        writer.close()
        await writer.wait_closed()


async def main() -> None:
    """HOST:PORT で TCP を待ち受け、接続ごとに handle_call() を起動して、止めるまで動き続ける。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("STT モデル読み込み中 %s", stt.MODEL)
    await asyncio.to_thread(stt.warmup)
    log.info("STT 準備完了")
    log.info("LLM 読み込み中 %s", llm.MODEL)
    await asyncio.to_thread(llm.warmup)
    log.info("LLM 準備完了")
    server = await asyncio.start_server(handle_call, HOST, PORT)
    log.info("AudioSocket 待ち受け %s:%d", HOST, PORT)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
