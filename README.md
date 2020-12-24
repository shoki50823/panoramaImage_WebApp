
# panoramaImage_WebApp
カメラで撮影した画像でパノラマ画像を生成するWEBアプリの試作


## ファイル構成
- cert: https通信を行うためのオレオレ証明書が格納されるディレクトリ
- post-images: 送信された画像が格納されるディレクトリ
- result-images: 生成されたパノラマ画像が格納されるディレクトリ
- static
    - loading.gif: ロード中のぐるぐるアニメーションgif
- views
    - *.tpl: HTMLテンプレートファイル
- panomara.py: パノラマ画像を生成するスクリプト
- server.py: サーバスクリプト
- start.bat: 仮想環境（Anaconda）起動からサーバスクリプトを実行するまでのバッチファイル


## 依存ライブラリ
- numpy
- bottle
- opencv-python

<br>

以下のコマンドを実行して，インストールする

```
$ pip install -r requirements.txt
```

## 使い方
### 前準備
opensslをインストールしていない人はインストールする必要があります．以下，参考資料です．
- windows:[WindowsにOpenSSLをインストールして証明書を取り扱う（基本編）](https://www.atmarkit.co.jp/ait/articles/1601/29/news043.html)

<br>


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
    - Anacondaを使っている人はバッチファイルも置いておきましたので，よかったら使ってください．
2. ブラウザからサーバスクリプトを動作させているマシンにアクセスする
    - https://192.168.1.XXX のように必ずhttpsでアクセスすること
    - WindowsではChrome，Macではsafariで動作確認済み．
3. ブラウザで開いたページを使って写真を撮って送信する
4. 複数枚写真をとったらボタンを押してパノラマ画像を生成し処理結果の画像を表示される．

## 備考
スマホのブラウザでアクセスする場合，カメラのプレビュー画面が表示されない．原因は不明．
