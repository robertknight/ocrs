#!/bin/sh

# Note: The long-term stability of these URLs is not guaranteed. If you are
# shipping a tool that uses Ocrs, bundle or host your own copy of the models.
DETECTION_MODEL="https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.onnx"
RECOGNITION_MODEL="https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.onnx"

curl "$DETECTION_MODEL" -o text-detection.onnx
curl "$RECOGNITION_MODEL" -o text-recognition.onnx
