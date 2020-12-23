<!DOCTYPE html>

<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>WebcamProcDemo</title>

    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css" integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS" crossorigin="anonymous">

    <style>
      html, body {
        width: 100%;
        height: 100%;
      }
      #vid {
        width: 100%;
        height: 100%;
      }
      #taken-img, #send-btn, #cancel-btn {
        display: none;
      }
      .shutter-btn-row {
        width: 100%;
        text-align: center;
      }
      #canvas {
        display: none;
      }
      .full-width {
        padding-left: 0;
        padding-right: 0;
        margin-left: 0;
        margin-right: 0;
      }
    </style>
  </head>

  <body>
    <div class="container-fluid">
      <div class="row">
        <div class="col-md-12 full-width">
          <video id="vid" autoplay class="img-fluid"></video>
          <img id="taken-img" class="img-fluid">
        </div>
        <div class="col-md-12">
          <div class="shutter-btn-row">
            <button id="take-btn" class="btn btn-primary btn-lg">Take</button>
            <button id="create-btn" class="btn btn-primary btn-lg">Create_Panorama</button>
            <button id="send-btn" class="btn btn-primary btn-lg">Send</button>
            <button id="cancel-btn" class="btn btn-secondary btn-lg">Cancel</button>
          </div>
        </div>
      </div>

      <canvas id="canvas"></canvas>

    </div>
  </body>

  <script>
    var vidStream = null;
    var vidElem = document.getElementById("vid");
    var canvasElem = document.getElementById("canvas");
    var takenImgElem = document.getElementById("taken-img");
    var takeBtnElem = document.getElementById("take-btn");
    var createBtnElem = document.getElementById("create-btn")
    var sendBtnElem = document.getElementById("send-btn");
    var cancelBtnElem = document.getElementById("cancel-btn");

    function startCamera() {
      navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "environment",  // 外側カメラ
            width: 1280,
            height: 1280},
          audio: false})
        .then((stream) => {
          vidStream = stream;
          vidElem.srcObject = vidStream;
        }).catch((error) => {
          window.alert("failed to getUserMedia: " + error);
        });
    }

    function takePicture() {
      var ctx = canvasElem.getContext("2d");
      var imgWidth = vidElem.videoWidth;
      var imgHeight = vidElem.videoHeight;
      canvasElem.setAttribute("width", imgWidth);
      canvasElem.setAttribute("height", imgHeight);
      ctx.drawImage(vidElem, 0, 0, imgWidth, imgHeight);
      takenImgElem.src = canvasElem.toDataURL("image/jpeg");
    }

    function sendPicture() {
      var dataURL = canvasElem.toDataURL("image/jpeg");
      var dataBase64 = dataURL.split(",")[1];
      var xhr = new XMLHttpRequest();
      xhr.addEventListener("loadend", (evt) => {
        if( xhr.status === 200 ) {
          // 画像を送信できたら結果ページへ遷移
          //画像を送信してもまたカメラページに戻る（改良後）
          var response = JSON.parse(xhr.responseText);
          location.href = response["index_page_url"];
        } else {
          window.alert("Failed to post image.");
        }
      });
      xhr.open("POST", "/post-image", true);
      xhr.send(dataBase64);
    }

    function createPanorama(){
      var xhr = new XMLHttpRequest();
      xhr.addEventListener("loadend", (evt) => {
        if( xhr.status === 200 ) {
          // 画像を送信できたら結果ページへ遷移
          var response = JSON.parse(xhr.responseText);
          location.href = response["result_page_url"];
        } else {
          window.alert("Failed to post image.");
        }
      });
      xhr.open("GET", "/panorama", true)
      xhr.send(null)

    }

    function toTakingMode() {
      vidElem.style.display = "block";
      takeBtnElem.style.display = "inline";
      takenImgElem.style.display = "none";
      createBtnElem.style.display = "inline";
      sendBtnElem.style.display = "none";
      cancelBtnElem.style.display = "none";
    }

    function toSendingMode() {
      vidElem.style.display = "none";
      takeBtnElem.style.display = "none";
      takenImgElem.style.display = "block";
      createBtnElem.style.display = "none";
      sendBtnElem.style.display = "inline";
      cancelBtnElem.style.display = "inline";
    }

    function init() {
      takeBtnElem.addEventListener("click", (evt) => {
        takePicture();
        toSendingMode();
      });

      createBtnElem.addEventListener("click", (evt) => {
        createPanorama();
      });

      sendBtnElem.addEventListener("click", (evt) => {
        sendPicture();
      });

      cancelBtnElem.addEventListener("click", (evt) => {
        toTakingMode();
      });

      startCamera();

    }
    init();

  </script>

</html>
