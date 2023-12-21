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

async function detectEyePositions() {
  await faceapi.nets.faceLandmark68Net.loadFromUri("/models");
  const video = document.getElementById("webcam");
  const detections = await faceapi.detectAllFaces(video).withFaceLandmarks();
}

var success = true;
var nowstatus = "";
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

    let message = "좋아요!";

    maxFaceSize = 240;
    minFaceSize = 170;
    threshold = 100;

    // 얼굴이 화면 중앙에 있는지 확인
    if (faceCenterX < screenCenterX - threshold) {
      message = "조금 더 오른쪽으로 이동해보세요.";
    } else if (faceCenterX + 60 > screenCenterX + threshold) {
      message = "조금 더 왼쪽으로 이동해보세요.";
    } else if (faceCenterY - 50 < screenCenterY - threshold) {
      message = "조금 더 아래쪽로 이동해보세요.";
    } else if (faceCenterY + 30 > screenCenterY + threshold) {
      message = "조금 더 위쪽으로 이동해보세요.";
    }

    // 얼굴이 카메라에 너무 가까운지 또는 멀리 있는지 확인
    if (face.width > maxFaceSize) {
      message = "너무 가까워요. 조금만 뒤로 물러나주세요.";
    } else if (face.width < minFaceSize) {
      message = "너무 멀어요. 카메라에 조금 더 가까이 다가와주세요.";
    }

    function showToast(position, icon, title, timer, toastStatus) {
      Swal.fire({
        position: position,
        icon: icon,
        title: title,
        showConfirmButton: false,
        timer: timer,
        toast: toastStatus,
      });
    }

    const statusMessages = {
      right: "조금 더 오른쪽으로 이동해보세요.",
      left: "조금 더 왼쪽으로 이동해보세요.",
      up: "조금 더 위쪽으로 이동해보세요.",
      down: "조금 더 아래쪽으로 이동해보세요.",
      back: "너무 가까워요. 조금만 뒤로 물러나주세요.",
      close: "너무 멀어요. 카메라에 조금 더 가까이 다가와주세요.",
      good: "좋아요!",
    };

    let newStatus = Object.keys(statusMessages).find(
      (key) => statusMessages[key] === message
    );

    if (newStatus && newStatus !== nowstatus) {
      if (message == "좋아요!") {
        trueFace = true;

        cheeseBtn.id = "clickable";
        cheeseBtn.innerHTML = "Go!";
        showToast("bottom-end", "success", message, 10000, true);
      } else {
        trueFace = false;
        cheeseBtn.innerHTML = "No!";
        cheeseBtn.removeAttribute("id");
        showToast("bottom-end", "info", message, 10000, true);
      }
      nowstatus = newStatus;
    }
  }
}, 100);

var trueFace = false;

document.addEventListener(
  "touchmove",
  function (event) {
    event.preventDefault();
  },
  { passive: false }
);

document.addEventListener("keydown", function (event) {
  if (
    [
      "ArrowUp",
      "ArrowDown",
      "Space",
      "PageUp",
      "PageDown",
      "Home",
      "End",
    ].includes(event.code)
  ) {
    event.preventDefault();
  }
});

document.addEventListener(
  "wheel",
  function (event) {
    if (event.ctrlKey) {
      event.preventDefault();
    }
  },
  { passive: false }
);

var cheeseBtn = document.querySelector(".cheese");
cheeseBtn.addEventListener("click", function () {
  if (trueFace) {
    sessionStorage.setItem("class",lastPredictedClass)
    sessionStorage.setItem("explain",lastexplain)

    window.location.href = "/pages/result.html";
  }
});

const URL = "/my_model/";

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
}

let animationFrameId;

async function loop() {
  webcam.update(); // 웹캠 프레임 업데이트
  await predict();
  animationFrameId = window.requestAnimationFrame(loop); // ID 저장
}

var lastPredictedClass = "";
var lastexplain = "";

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
  lastPredictedClass = predictedClass;
  lastexplain = explain;
}