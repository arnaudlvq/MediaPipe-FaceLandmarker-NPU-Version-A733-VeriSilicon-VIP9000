#!/usr/bin/env bash
# Download Google's MediaPipe FaceLandmarker bundle and extract the 3 TFLite
# models that this project converts. They are NOT committed here (fetch them).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
echo "Downloading face_landmarker.task ..."
curl -L -o models/face_landmarker.task "$URL"
echo "Extracting TFLite models ..."
( cd models && unzip -o face_landmarker.task \
    face_detector.tflite face_landmarks_detector.tflite face_blendshapes.tflite )
echo "Done -> models/"
