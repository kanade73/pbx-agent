# pbx-agent

Asterisk とローカル音声処理（STT / LLM / TTS）を組み合わせた、オフライン動作を前提とした音声エージェントの実験用リポジトリです。

## ディレクトリ構成

- `asterisk/`
  - Docker Compose で起動する Asterisk 設定
  - 内線 `100` は AudioSocket で voice-agent（`host.docker.internal:9090`）へ転送
- `voice-agent/`
  - AudioSocket サーバー本体（Python）
  - 受信音声を STT → LLM → TTS の流れで処理して返答

## 使い方（最小手順）

1. Asterisk の環境ファイルを作成
   - `asterisk/.env.example` をコピーして `asterisk/.env` を作成
   - `LAN_IP` などを実環境に合わせて設定
2. Asterisk を起動
   - `asterisk/` で `docker compose up -d`
3. voice-agent を起動
   - `voice-agent/` で依存関係をインストール後、`python -m voice_agent` を実行
4. SIP クライアントから内線 `100` へ発信して動作確認

## ライセンス

MIT