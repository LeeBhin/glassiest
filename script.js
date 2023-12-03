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
  const canvas = faceapi.createCanvasFromMedia(video);
  document.querySelector(".zonewrap").append(canvas);
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);
  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    // faceapi.draw.drawDetections(canvas, resizedDetections);
    // faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    // faceapi.draw.drawFaceExpressions(canvas, resizedDetections)
    resizedDetections.forEach((detection) => {
      const leftEye = detection.landmarks.getLeftEye();
      const rightEye = detection.landmarks.getRightEye();

      // 눈의 위치 출력
      drawEyePosition(rightEye);
    });
  }, 10);
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
  webcam = new tmImage.Webcam(10, 10, flip); // width, height, flip
  await webcam.setup(); // request access to the webcam
  await webcam.play();
  window.requestAnimationFrame(loop);

  // append elements to the DOM
  document.getElementById("webcam-container").appendChild(webcam.canvas);
  labelContainer = document.getElementById("label-container");
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

  // 클래스에 따라 결과 형식 변환
  let formattedResult = "";
  switch (predictedClass) {
    case "각진":
      formattedResult = "각진형";
      break;
    case "둥근":
      formattedResult = "둥근형";
      break;
    case "긴":
      formattedResult = "긴형";
      break;
    // 추가적인 클래스에 대한 처리도 필요하다면 여기에 추가
    default:
      formattedResult = predictedClass;
  }

  // 확률이 가장 높은 클래스 및 변환된 결과 표시
  const classPrediction = formattedResult;
  labelContainer.innerHTML = classPrediction;
}

async function detectEyePositions() {
  await faceapi.nets.faceLandmark68Net.loadFromUri("/models");
  const video = document.getElementById("webcam");
  const detections = await faceapi.detectAllFaces(video).withFaceLandmarks();

  detections.forEach((detection) => {
    const landmarks = detection.landmarks;
    const leftEye = landmarks.getLeftEye();
    const rightEye = landmarks.getRightEye();

    console.log("Left Eye Position:", leftEye);
    console.log("Right Eye Position:", rightEye);
  });
}

// 이미지를 미리 로드합니다.
let eyeImage = new Image();
eyeImage.src = "pic.png"; // 원하는 이미지의 경로를 지정하세요.

eyeImage.onload = function () {
  // 이미지가 로드되면 크기를 설정합니다.
  eyeImage.width = 220; // 원하는 너비
  eyeImage.height = 90; // 원하는 높이
};

// 눈의 위치에 이미지를 그리는 함수
function drawEyePosition(eye) {
  const canvas = document.querySelector("canvas");
  const context = canvas.getContext("2d");

  // 이미지의 중심이 눈의 위치에 오도록 조정
  const eyeX = eye[0].x - eyeImage.width + 90;
  const eyeY = eye[0].y - eyeImage.height / 2 + 20;

  // 이미지 그리기
  context.drawImage(eyeImage, eyeX, eyeY, eyeImage.width, eyeImage.height);
}
