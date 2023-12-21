var preclass = sessionStorage.getItem("class")
var explain = sessionStorage.getItem("explain")

document.getElementById('faceType').innerHTML = preclass
document.getElementById('explain').innerHTML = explain


const video = document.getElementById("video");

Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceExpressionNet.loadFromUri("/models"),
]).then(startVideo);

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    (stream) => (video.srcObject = stream),
    (err) => console.error(err)
  );
}

video.addEventListener("play", () => {
  // 기존 캔버스 (이미지용)
  const canvas = faceapi.createCanvasFromMedia(video);
  document.querySelector(".zonewrap").append(canvas);

  // 새 캔버스 생성 (랜드마크용)
  const landmarksCanvas = faceapi.createCanvasFromMedia(video);
  document.querySelector(".faceLandmarks").append(landmarksCanvas);

  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);
  faceapi.matchDimensions(landmarksCanvas, displaySize);

  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    // 기존 캔버스에서 이미지 그리기 전에 캔버스 초기화
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    // faceapi.draw.drawDetections(canvas, resizedDetections);
    // faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    // faceapi.draw.drawFaceExpressions(canvas, resizedDetections)

    resizedDetections.forEach((detection) => {
      const rightEye = detection.landmarks.getRightEye();
      const leftEye = detection.landmarks.getLeftEye();
      const faceWidth = detection.detection.box.width / 1.2;
      const faceHeight = detection.detection.box.height / 3.5;

      // 눈의 위치와 얼굴 크기에 맞게 이미지 조절
      drawEyePosition(leftEye, rightEye, faceWidth, faceHeight);
    });

    // 새 캔버스에서 랜드마크만 그리기
    landmarksCanvas
      .getContext("2d")
      .clearRect(0, 0, landmarksCanvas.width, landmarksCanvas.height);
    // faceapi.draw.drawFaceLandmarks(landmarksCanvas, resizedDetections);
  }, 0);
});

let eyeImage = new Image();

function imgselect(gsNum) {
  // 이미지를 미리 로드합니다.
  eyeImage.src = `/pictures/${gsNum}.png`; // 원하는 이미지의 경로를 지정하세요.

  eyeImage.onload = function () {
    // 이미지가 로드되면 크기를 설정합니다.
    eyeImage.width = 220; // 원하는 너비
    eyeImage.height = 90; // 원하는 높이
  };
}

imgselect("각진형/brow1");



function drawEyePosition(leftEye, rightEye, faceWidth, faceHeight) {
  const canvas = document.querySelector(".zonewrap canvas");
  const context = canvas.getContext("2d");

  // 스케일 계수 조정
  const scale = 1.0;
  const scaleWidth = (faceWidth / eyeImage.width) * scale;
  const scaleHeight = (faceHeight / eyeImage.height) * scale;
  const scaledWidth = eyeImage.width * scaleWidth;
  const scaledHeight = eyeImage.height * scaleHeight;

  // 왼쪽 눈과 오른쪽 눈의 중심 좌표 계산
  const leftEyeCenter = getEyeCenter(leftEye);
  const rightEyeCenter = getEyeCenter(rightEye);

  // 양쪽 눈의 중간 지점 계산
  const eyesCenterX = (leftEyeCenter.x + rightEyeCenter.x) / 2;
  const eyesCenterY = (leftEyeCenter.y + rightEyeCenter.y) / 1.95;

  // 이미지가 양쪽 눈의 중간에 오도록 조정
  const eyeX = eyesCenterX - scaledWidth / 2;
  const eyeY = eyesCenterY - scaledHeight / 2;

  // 얼굴 기울기 계산
  const angle = Math.atan2(
    rightEye[0].y - leftEye[0].y,
    rightEye[0].x - leftEye[0].x
  );

  // 캔버스에 이미지를 그리기 전에 캔버스 상태 저장
  context.save();

  // 캔버스 중심을 이미지의 중심으로 이동
  context.translate(eyesCenterX, eyesCenterY);

  // 이미지 회전
  context.rotate(angle);

  // 이미지를 원래 위치로 되돌리고 그리기
  context.drawImage(
    eyeImage,
    -scaledWidth / 2,
    -scaledHeight / 2,
    scaledWidth,
    scaledHeight
  );

  // 캔버스 상태를 이전 상태로 복원
  context.restore();
}

function getEyeCenter(eye) {
  let centerX = 0;
  let centerY = 0;
  eye.forEach((point) => {
    centerX += point.x;
    centerY += point.y;
  });
  return { x: centerX / eye.length, y: centerY / eye.length };
}
