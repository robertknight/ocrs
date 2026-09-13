#!/bin/sh

DETECTION_MODEL="https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.onnx"
RECOGNITION_MODEL="https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.onnx"

curl "$DETECTION_MODEL" -o text-detection.onnx
curl "$RECOGNITION_MODEL" -o text-recognition.onnx
