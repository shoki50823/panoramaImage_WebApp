<!DOCTYPE html>

<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>WebcamProcDemo - Result</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css" integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS" crossorigin="anonymous">

    <style>
      html, body {
        width: 100%;
        height: 100%;
      }
      .full-width {
        padding-left: 0;
        padding-right: 0;
        margin-left: 0;
        margin-right: 0;
        width: 100%;
      }
      .centering {
        text-align: center;
      }
    </style>
  </head>

  <body>
    <div class="container-fluid">
      <div class="row">
        <div class="col-md-12 full-width">
          <h1>Result</h1>
        </div>
        <div class="col-md-12 full-width centering">
          <img id="result-img" class="img-fluid" src="/static/loading.gif">
        </div>
    </div>
  </body>

  <script>
    var resultImgElem = document.getElementById("result-img");
    var timeout = 30;  // タイムアウト秒数
    var waitingTime = 0;

    function isResultExists(resultId) {
      return new Promise((resolve, reject) => {
        var xhr = new XMLHttpRequest();
        xhr.open("GET", "/is-result-exists/" + resultId, true);
        xhr.addEventListener("loadend", (evt) => {
          if( xhr.status === 200 ) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            reject();
          }
        });
        xhr.send(null);
      });
    }

    function checkResultExistsLoop(resultId) {
      waitingTime++;
      if( timeout < waitingTime ) {
        // タイムアウトしたらこれ以上結果を問い合わせない
        window.alert("Process takes too long time. Please retry.");
        return;
      }

      isResultExists(resultId)
        .then((response) => {
          if( response["exists"] === "true" ) {
            // 処理結果の画像があったら表示
            resultImgElem.src = response["result_image_path"];
          } else {
            // 処理が終わってなければ，1秒後に再度結果を問い合わせる
            setTimeout(() => checkResultExistsLoop(resultId), 1000);
          }
        }).catch(() => {
          window.alert("Failed to isResultExists.")
        });
    }
    checkResultExistsLoop("{{ result_id }}");

  </script>

</html>
