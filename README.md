# pbx-agent

Asterisk（PBX）とローカルの音声処理（STT / LLM / TTS）を組み合わせた、電話で話せる音声エージェントの実験用リポジトリです。初回セットアップでモデルを取得したあとは、ネットワークなしで動作します。

## 動作環境

- macOS（Apple Silicon / Intel）
  - 音声合成に macOS 標準の `say` コマンド（声: Kyoko）を使うため、Linux / Windows では応答音声が出ません
- Docker Desktop（コンテナから `host.docker.internal` でホストに届くこと）
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)（LLM。モデルは `qwen3:8b`）
- SIP クライアント（Zoiper など）

## ディレクトリ構成

- `asterisk/`
  - Docker Compose で起動する Asterisk 設定
  - 内線 `100` は AudioSocket で voice-agent（`host.docker.internal:9090`）へ転送
  - 内線 `600` はエコーテスト、`6001` / `6002` は SIP クライアント用の内線
- `voice-agent/`
  - AudioSocket サーバー本体（Python）
  - 受信音声を STT（Whisper）→ LLM（Ollama）→ TTS（`say`）の流れで処理して返答

## 使い方（最小手順）

1. Asterisk の設定ファイルを作成
   - `asterisk/.env.example` をコピーして `asterisk/.env` を作成し、`LAN_IP` を Mac の LAN IP（`ipconfig getifaddr en0` で確認）に書き換える
   - `asterisk/pjsip_auth.conf.example` をコピーして `asterisk/pjsip_auth.conf` を作成し、`password` を書き換える
     - 作成前に Asterisk を起動すると、Docker が同名の空ディレクトリを作ってしまう（Asterisk はエラーを出さずに起動するが、SIP 登録が通らない）。その場合は削除してから作り直す
2. Asterisk を起動
   - `asterisk/` で `docker compose up -d`
3. LLM と TTS を準備
   - Ollama を起動し、`ollama pull qwen3:8b` でモデルを取得
   - `say -v Kyoko こんにちは` で読み上げられることを確認（声がなければシステム設定の「読み上げコンテンツ」から Kyoko を追加）
4. voice-agent を起動
   - `voice-agent/` で実行
     - Apple Silicon Mac: `uv sync && uv run python -m voice_agent`
     - Intel Mac: `uv sync --extra faster && uv run python -m voice_agent`
   - 初回は Whisper のモデルを Hugging Face から取得する（`~/.cache/huggingface` に保存）。2 回目以降は `HF_HUB_OFFLINE=1` を付けて起動すればネットワークに接続しない
   - ログに `AudioSocket 待ち受け 0.0.0.0:9090` が出れば準備完了
5. SIP クライアントを登録して発信
   - ユーザー名は `6001`（または `6002`）、パスワードは `pjsip_auth.conf` の値、サーバーは `LAN_IP`（`SIP_PORT` を 5060 以外にした場合は `LAN_IP:SIP_PORT`）、トランスポートは UDP
   - まず内線 `600`（エコーテスト）に発信し、自分の声が返ってくることを確認
   - 内線 `100` に発信し、話しかけると音声エージェントが応答する

## ライセンス

MIT
