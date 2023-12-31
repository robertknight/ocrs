import { readFile } from "fs/promises";

import { program } from "commander";
import sharp from "sharp";

import {
  OcrEngine,
  OcrEngineInit,
  default as initOcrLib,
} from "../../dist/ocrs.js";

/**
 * Load a JPEG or PNG image from `path` and return the RGB image data as an
 * `ImageData`-like object.
 */
async function loadImage(path) {
  const image = await sharp(path);
  const { width, height } = await image.metadata();
  const data = await image.raw().toBuffer();
  return {
    data: new Uint8Array(data),
    width,
    height,
  };
}

/**
 * Detect text in an image and return the result as a JSON-serialiable object.
 *
 * @param {OcrEngine} ocrEngine
 * @param {OcrInput} ocrInput
 */
function detectText(ocrEngine, ocrInput) {
  const detectedLines = ocrEngine.detectText(ocrInput);
  const lines = detectedLines.map((line) => {
    const words = line.words().map((rect) => {
      return {
        rect: Array.from(rect.boundingRect()),
      };
    });

    return {
      words,
    };
  });
  return {
    lines,
  };
}

/**
 * Detect and recognize text in an image and return the result as a
 * JSON-serialiable object.
 *
 * @param {OcrEngine} ocrEngine
 * @param {OcrInput} ocrInput
 */
function detectAndRecognizeText(ocrEngine, ocrInput) {
  const textLines = ocrEngine.getTextLines(ocrInput);
  const lines = textLines.map((line) => {
    const words = line.words().map((word) => {
      return {
        text: word.text(),
        rect: Array.from(word.rotatedRect().boundingRect()),
      };
    });

    return {
      text: line.text(),
      words,
    };
  });
  return {
    lines,
  };
}

program
  .name("ocr")
  .argument("<detection_model>", "Text detection model path")
  .argument("<recognition_model>", "Text recognition model path")
  .argument("<image>", "Input image path")
  .option("-d, --detect-only", "Detect text, but don't recognize it")
  .option("-j, --json", "Output JSON")
  .action(
    async (detectionModelPath, recognitionModelPath, imagePath, options) => {
      // Concurrently load the OCR library, text detection and recognition models,
      // and input image.
      const [_, detectionModel, recognitionModel, image] = await Promise.all([
        readFile("dist/ocrs_bg.wasm").then(initOcrLib),
        readFile(detectionModelPath).then((data) => new Uint8Array(data)),
        readFile(recognitionModelPath).then((data) => new Uint8Array(data)),
        loadImage(imagePath),
      ]);

      const ocrInit = new OcrEngineInit();
      ocrInit.setDetectionModel(detectionModel);

      // TODO - Don't require the recognition model when doing only detection.
      ocrInit.setRecognitionModel(recognitionModel);

      const ocrEngine = new OcrEngine(ocrInit);
      const ocrInput = ocrEngine.loadImage(
        image.width,
        image.height,
        image.data
      );

      if (options.detectOnly) {
        const json = detectText(ocrEngine, ocrInput);
        console.log(JSON.stringify(json, null, 2));
      } else if (options.json) {
        const json = detectAndRecognizeText(ocrEngine, ocrInput);
        console.log(JSON.stringify(json, null, 2));
      } else {
        const text = ocrEngine.getText(ocrInput);
        console.log(text);
      }
    }
  )
  .parse();
