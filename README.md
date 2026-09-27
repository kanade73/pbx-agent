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
   - `asterisk/pjsip_auth.conf` を作成（`.gitignore` 済み。`6001` と `6002` の `type=auth` / `auth_type=userpass` / `username` / パスワードを設定）
2. Linux で Docker Engine を使う場合のみ `host.docker.internal` を解決できるようにする
   - `asterisk/compose.yaml` の `services.asterisk` に以下を追加
     ```yaml
     extra_hosts:
       - "host.docker.internal:host-gateway"
     ```
3. Asterisk を起動
   - `asterisk/` で `docker compose up -d`
4. voice-agent を起動
   - `voice-agent/` で実行
     - Apple Silicon Mac: `uv sync && uv run python -m voice_agent`
     - それ以外（Linux / Windows / Intel Mac）: `uv sync --extra faster && uv run python -m voice_agent`
5. SIP クライアントから内線 `100` へ発信して動作確認

## ライセンス

MIT