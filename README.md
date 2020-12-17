
# anoramaImage_WebApp
スマホで撮った写真でパノラマ画像を生成するWEBアプリの試作


## ファイル構成
- cert: https通信を行うためのオレオレ証明書が格納されるディレクトリ
- post-images: スマホから送信された画像が格納されるディレクトリ
- result-images: 処理済みの画像が格納されるディレクトリ
- static
    - loading.gif: ロード中のぐるぐるアニメーションgif
- views
    - *.tpl: HTMLテンプレートファイル
- panomara.py: パノラマ画像を生成するスクリプト
- server.py: サーバスクリプト


## 依存ライブラリ
- bottle
- opencv-python


## 使い方
### 前準備
https通信を行うために，オレオレ証明書を生成する．
以下のコマンドを実行して，certディレクトリに秘密鍵と公開鍵を生成する．

```
$ mkdir cert
$ cd cert
$ openssl genrsa 2048 > server.key
$ openssl req -new -key server.key > server.csr
$ openssl x509 -days 365 -req -signkey server.key < server.csr > server.crt
```


### 動かし方
1. サーバスクリプトserver.pyを起動する
    - 使っているウイルス対策ソフトによっては，ファイアウォールで443/tcpに穴を開ける必要があるかも
2. スマホのブラウザ（Chromeのみ動作確認済み）からサーバスクリプトを動作させているマシンにアクセスする
    - https://192.168.1.XXX のように必ずhttpsでアクセスすること
3. ブラウザで開いたページを使って写真を撮って送信する
4. 複数枚写真をとったらボタンを押してパノラマ画像を生成し処理結果の画像を表示


