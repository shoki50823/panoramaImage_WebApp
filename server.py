
from wsgiref.simple_server import make_server
import ssl
import base64
from pathlib import Path
import json
from datetime import datetime
import subprocess
import bottle
import os
import glob
import shutil

from panorama import *


def generate_id():
    """
    現在時刻を用いて固有のIDを生成する

    Returns:
        str: 固有のID
    """
    return datetime.now().strftime("D%Y-%m-%d-T%H-%M-%S")


@bottle.route("/static/<path:path>")
def static_files(path):
    """ 静的ファイル配信 """
    return bottle.static_file(path, "./static")


@bottle.route("/result-images/<img_path>")
def result_images(img_path):
    """ 処理済み画像配信 """
    return bottle.static_file(img_path, "./result-images")


@bottle.route("/")
def root():

    """ ルートページ配信 """
    return bottle.template("index")


@bottle.route("/result/<result_id>")
def result_page(result_id):
    """ 渡されたIDの処理結果ページ配信 """
    return bottle.template("result", result_id=result_id)


@bottle.route("/post-image", method="POST")
def post_image():
    """
    画像アップロード
    POSTの本文として送信される，base64エンコード済みの画像画像ファイルを受け取る．
    レスポンスとして，再びカメラで撮影を行うためにルートページ返す．
    """

    date_id = generate_id()
    save_path = Path("post-images") / (date_id + ".jpg")

    # 送られてきた画像を保存
    data_b64 = bottle.request.body.read().decode("ascii")
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(data_b64))

    return json.dumps({
        # "result_id": result_id,
        # "result_page_url": "/result/" + result_id
        "index_page_url": "/" 
    })


@bottle.route("/panorama", method="GET")
def panorama():

    result_id = generate_id()
    result_path = Path("result-images") / (result_id + ".jpg")

    # #　post-images内の全ての画像を読み取る
    image_names = []
    img_path = glob.glob("post-images/*")
    for fname in img_path:
        image_names.append(fname)

    # パノラマ関数を実行
    panorama_main(image_names, result_path)

    # post-imagesを空にする
    # 空にしないと次に作成するときに前回撮影した画像も一緒に使用される
    shutil.rmtree("post-images")
    os.mkdir("post-images")

    return json.dumps({
        "result_id": result_id,
        "result_page_url": "/result/" + result_id
    })



@bottle.route("/is-result-exists/<result_id>")
def is_result_exists(result_id):
    """ 渡されたIDの処理結果が存在しているかどうかを取得する """
    result_path = Path("result-images") / (result_id + ".jpg")
    print(result_path)

    return json.dumps({
        "exists": "true" if result_path.exists() else "false",
        "result_image_path": "/" + str(result_path),
        "result_id": result_id
    })


def main():
    with make_server("0.0.0.0", 443, bottle.default_app()) as httpd:
        httpd.socket = ssl.wrap_socket(
            httpd.socket,
            server_side=True,
            certfile="cert/server.crt",
            keyfile="cert/server.key")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
