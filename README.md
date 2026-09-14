# MiRAI Server for Whisper

構築済みのローカル [openai-whisper](https://github.com/openai/whisper) 環境を、MiRAI Headから利用できるOpenAI互換のHTTP文字起こしサーバーとして起動します。

Windows向けです。Whisper、PyTorch、FFmpegおよび使用するWhisperモデルが正常に動作する環境をあらかじめ用意してください。本リポジトリにはWhisper本体やモデルファイルは含まれません。

## 特長

- 既存のWhisper環境へ3ファイルをコピーしてBATをダブルクリックするだけで起動
- `env`、`.venv`、`venv`などの一般的な仮想環境を自動検出
- FastAPIなど、サーバー用の追加パッケージは初回起動時に自動導入
- `POST /v1/audio/transcriptions` のOpenAI互換エンドポイント
- CUDAが利用できる場合は自動的にGPUを使用
- 家庭内LANから利用できるよう、既定で全IPv4インターフェースのポート8000を使用
- 無音・非音声・極端に短い音声や、明らかな反復結果を抑制
- 言語別の固有名詞辞書に対応

## 前提条件

- Windows 10またはWindows 11
- Python 3.10以降を推奨
- `openai-whisper` とPyTorchが導入済みであること
- FFmpegがインストール済みで、Whisperから利用できること
- 初回のサーバー依存パッケージ導入時にインターネットへ接続できること

既存環境で次のコマンドが成功することを確認してください。

```bat
path\to\python.exe -c "import whisper, torch; print(whisper.available_models())"
```

## インストール

1. GitHubの **Code → Download ZIP** からダウンロードして展開します。
2. 次の3ファイルを既存Whisper環境のルートへコピーします。

```text
start_mirai_whisper_server.bat
mirai_whisper_server.py
requirements-server.txt
```

推奨配置例:

```text
C:\Whisper-ASR\
├─ env\
│  └─ Scripts\python.exe
├─ models\                       （任意）
│  └─ large-v3-turbo.pt          （任意）
├─ start_mirai_whisper_server.bat
├─ mirai_whisper_server.py
└─ requirements-server.txt
```

次の場所から、Whisperをimportできる最初のPythonが選択されます。

1. 環境変数 `MIRAI_WHISPER_PYTHON`
2. 環境変数 `WHISPER_ASR_PYTHON`
3. BATと同じ場所の `env\Scripts\python.exe`
4. BATと同じ場所の `.venv\Scripts\python.exe`
5. BATと同じ場所の `venv\Scripts\python.exe`
6. BATと同じ場所の `Scripts\python.exe`
7. `PATH` 上の `python.exe`

仮想環境が別の場所にある場合は、Windowsのユーザー環境変数 `MIRAI_WHISPER_PYTHON` に対象の `python.exe` の絶対パスを設定してください。

## 起動

`start_mirai_whisper_server.bat`をダブルクリックします。

初回だけFastAPI、Uvicorn、multipart受信用ライブラリ、WebRTC VADが既存Python環境へ追加されます。その後Whisperモデルが読み込まれ、次の表示になれば準備完了です。

```text
Whisper model 'large-v3-turbo' is ready.
Uvicorn running on http://0.0.0.0:8000
```

`0.0.0.0`は待受設定であり、MiRAI Headへ入力する接続先アドレスではありません。

## MiRAI Headから接続する

コマンドプロンプトで次を実行し、PCのIPv4アドレスを確認します。

```bat
ipconfig
```

たとえばPCのIPv4アドレスが `192.168.0.15` なら、MiRAI Headには次のURLを設定します。

```text
http://192.168.0.15:8000
```

Windows Firewallの確認画面が表示された場合は、使用している**プライベートネットワーク**のみ受信を許可してください。本サーバーは家庭内LANでの利用を想定しており、既定ではAPIキーを要求しません。インターネットへ直接公開しないでください。

## 動作確認

サーバーと同じPCからヘルスチェックできます。

```bat
curl.exe --fail-with-body http://127.0.0.1:8000/health
```

成功例:

```json
{"status":"ok","model":"large-v3-turbo","device":"cuda","glossary_terms":0,"glossary_file":""}
```

音声ファイルを送信する例:

```bat
curl.exe --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions ^
  -F "file=@C:\path\speech.wav;type=audio/wav" ^
  -F "model=large-v3-turbo" ^
  -F "language=ja" ^
  -F "response_format=json"
```

成功時は次の形式で返ります。

```json
{"text":"認識された文章"}
```

`response_format`は`json`、`text`、`verbose_json`に対応します。`language=auto`または言語を省略すると自動判定します。

## モデルの保存場所

モデルフォルダーは次の優先順位で選択されます。

1. 環境変数 `MIRAI_WHISPER_MODEL_DIR`
2. 環境変数 `WHISPER_ASR_MODELS`
3. BATと同じ場所の `models` フォルダー
4. Whisper標準キャッシュの `%USERPROFILE%\.cache\whisper`

未配置の公式モデル名を指定した場合は、選択されたモデルフォルダーへWhisperがモデルをダウンロードすることがあります。

既定モデルは`large-v3-turbo`です。別のモデルで起動する場合はコマンドプロンプトから指定できます。

```bat
start_mirai_whisper_server.bat --model medium
```

## 固有名詞辞書

BATと同じ場所に`glossary.txt`、または言語別の`glossary.<言語コード>.txt`を作成すると、各行がWhisperの初期プロンプトへ追加されます。UTF-8で、1行に1語記述してください。

```text
MiRAI Head
TextMeshPro
Magica Cloth 2
```

日本語では`glossary.ja.txt`、英語では`glossary.en.txt`が優先されます。ファイルはリクエストごとに更新確認されるため、編集後のサーバー再起動は不要です。

## 主な起動オプション

```text
--host HOST
--port PORT
--model MODEL
--model-dir PATH
--device auto|cuda|cpu
--max-upload-mib SIZE
--glossary PATH
```

例:

```bat
start_mirai_whisper_server.bat --port 8080 --device cpu
```

## トラブルシューティング

### Python環境が見つからない

```text
[ERROR] A Python environment containing openai-whisper was not found.
```

3ファイルを`env`や`.venv`と同じ場所へ配置したことを確認してください。環境が別の場所なら、`MIRAI_WHISPER_PYTHON`へ使用する`python.exe`を設定します。

### サーバー用パッケージをインストールできない

インターネット接続と、対象Python環境で`pip`が利用できることを確認してください。手動で導入する場合:

```bat
path\to\python.exe -m pip install -r requirements-server.txt
```

### MiRAI Headから接続できない

- MiRAI Headに`127.0.0.1`や`0.0.0.0`ではなく、PCのLAN IPv4アドレスを設定したか確認します。
- PCとMiRAI Headが同じ家庭内LANへ接続されているか確認します。
- Windows FirewallでプライベートネットワークからのPython受信が許可されているか確認します。
- PCのIPv4アドレスが変わっていないか、`ipconfig`でもう一度確認します。

### ポート8000を使用できない

別のポートで起動し、MiRAI Head側のURLも同じポートへ変更します。

```bat
start_mirai_whisper_server.bat --port 8080
```

## 停止

サーバーのウィンドウで `Ctrl+C` を押すか、ウィンドウを閉じます。
