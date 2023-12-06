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
      const faceHeight = detection.detection.box.height / 3;

      // 눈의 위치와 얼굴 크기에 맞게 이미지 조절
      drawEyePosition(leftEye, rightEye, faceWidth, faceHeight);
    });

    // 새 캔버스에서 랜드마크만 그리기
    landmarksCanvas
      .getContext("2d")
      .clearRect(0, 0, landmarksCanvas.width, landmarksCanvas.height);
    faceapi.draw.drawFaceLandmarks(landmarksCanvas, resizedDetections);
  }, 0);
});

// More API functions here:
// https://github.com/googlecreativelab/teachablemachine-community/tree/master/libraries/image

// the link to your model provided by Teachable Machine export panel
const URL = "./my_model/";

let model, webcam, labelContainer, maxPredictions;

init();

// Load the image model and setup the webcam
async function init() {
  const modelURL = URL + "model.json";
  const metadataURL = URL + "metadata.json";

  // load the model and metadata
  // Refer to tmImage.loadFromFiles() in the API to support files from a file picker
  // or files from your local hard drive
  // Note: the pose library adds "tmImage" object to your window (window.tmImage)
  model = await tmImage.load(modelURL, metadataURL);
  maxPredictions = model.getTotalClasses();

  // Convenience function to setup a webcam
  const flip = true; // whether to flip the webcam
  webcam = new tmImage.Webcam(200, 200, flip); // width, height, flip
  await webcam.setup(); // request access to the webcam
  await webcam.play();
  window.requestAnimationFrame(loop);

  // append elements to the DOM
  document.getElementById("webcam-container").appendChild(webcam.canvas);
  labelContainer = document.getElementById("faceType");
  for (let i = 0; i < maxPredictions; i++) {
    // and class labels
    labelContainer.appendChild(document.createElement("div"));
  }
}

async function loop() {
  webcam.update(); // update the webcam frame
  await predict();
  window.requestAnimationFrame(loop);
}

async function predict() {
  // 이미지 모델에 웹캠 이미지를 입력하여 예측
  const prediction = await model.predict(webcam.canvas);

  // 가장 높은 확률을 가진 클래스 찾기
  let maxProbability = 0;
  let predictedClass = "";

  for (let i = 0; i < maxPredictions; i++) {
    const currentProbability = prediction[i].probability;

    if (currentProbability > maxProbability) {
      maxProbability = currentProbability;
      predictedClass = prediction[i].className;
    }
  }

  let explain = "";
  switch (predictedClass) {
    case "각진형":
      explain = `각진 얼굴형은 눈에 띄는 각진 턱과 넓은 이마가 특징적입니다. 이마, 볼, 턱선의 너비가 거의 비슷해 얼굴 전체적으로 강인한 인상을 줍니다. 직선적인 턱선은 이 얼굴형의 가장 두드러진 특징으로, 균형 잡힌 각진 모양을 보여줍니다.`;
      break;
    case "둥근형":
      explain = `둥근 얼굴형은 부드러운 윤곽과 둥근 턱이 특징입니다. 얼굴 길이와 너비가 비슷하며, 턱선이 부드럽게 곡선을 그려 볼이 도드라져 보일 수 있습니다. 이 얼굴형은 친근하고 부드러운 인상을 주는 것으로 알려져 있습니다.`;
      break;
    case "긴형":
      explain = `긴 얼굴형은 전체적으로 길쭉한 형태를 가지며, 이마, 볼, 턱의 너비가 비슷하게 나타납니다. 이 얼굴형은 종종 좁고 긴 이마와 턱을 가지고 있으며, 턱선이 덜 두드러질 수 있습니다. 긴 얼굴형은 세로 길이가 강조되는 특징을 가지고 있습니다.`;
      break;
    case "달걀형":
      explain = `달걀형 얼굴은 이마가 넓고 턱이 좁아지며, 전반적으로 균형 잡힌 비율을 가지고 있습니다. 얼굴 길이가 너비보다 길고, 턱선은 부드러운 곡선을 이룹니다. 이 얼굴형은 다양한 헤어스타일과 메이크업이 잘 어울리는 것으로 알려져 있습니다.`;
      break;
    case "하트형":
      explain = `하트형 얼굴은 이마가 넓고 턱이 좁아지는 모양으로, 맨 위가 넓고 아래로 갈수록 좁아지는 하트 모양을 연상시킵니다. 특히 턱이 뾰족하게 나타나는 경우가 많으며, 볼은 상대적으로 더 둥근 형태를 보입니다. 이 얼굴형은 이마의 너비가 눈에 띄며, 턱선이 좁아지는 것이 특징입니다.`;
      break;
  }

  labelContainer.innerText = predictedClass;
  document.getElementById("explain").innerText = explain;
}

async function detectEyePositions() {
  await faceapi.nets.faceLandmark68Net.loadFromUri("/models");
  const video = document.getElementById("webcam");
  const detections = await faceapi.detectAllFaces(video).withFaceLandmarks();
}

// 이미지를 미리 로드합니다.
let eyeImage = new Image();
eyeImage.src = "pic.png"; // 원하는 이미지의 경로를 지정하세요.

eyeImage.onload = function () {
  // 이미지가 로드되면 크기를 설정합니다.
  eyeImage.width = 220; // 원하는 너비
  eyeImage.height = 90; // 원하는 높이
};

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
  const eyesCenterY = (leftEyeCenter.y + rightEyeCenter.y) / 2;

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

setInterval(async () => {
  const detections = await faceapi
    .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();

  if (detections.length > 0) {
    const face = detections[0].detection.box;
    const screenCenterX = video.width / 2;
    const screenCenterY = video.height / 2;
    const faceCenterX = face.x + face.width / 2;
    const faceCenterY = face.y + face.height / 2;

    let message = "";

    maxFaceSize = 200;
    minFaceSize = 100;
    threshold = 100;

    // 얼굴이 화면 중앙에 있는지 확인
    if (faceCenterX < screenCenterX - threshold) {
      message = "조금 더 오른쪽으로 이동해보세요.";
    } else if (faceCenterX > screenCenterX + threshold) {
      message = "조금 더 왼쪽으로 이동해보세요.";
    } else if (faceCenterY < screenCenterY - threshold) {
      message = "조금 더 아래쪽로 이동해보세요.";
    } else if (faceCenterY > screenCenterY + threshold) {
      message = "조금 더 위쪽으로 이동해보세요.";
    }

    // 얼굴이 카메라에 너무 가까운지 또는 멀리 있는지 확인
    if (face.width > maxFaceSize) {
      message = "너무 가까워요. 조금만 뒤로 물러나주세요.";
    } else if (face.width < minFaceSize) {
      message = "너무 멀어요. 카메라에 조금 더 가까이 다가와주세요.";
    }

    // 메시지 표시
    if (!message == "") {
      alert(message);
    }
  }
}, 100);
