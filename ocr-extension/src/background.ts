import {
  DetectedLineList,
  OcrEngine,
  OcrEngineInit,
  default as initOcrLib,
} from "../build/wasnn_ocr.js";
import type { LineRecResult, RotatedRect, WordRecResult } from "./types";
import type * as contentModule from "./content";
import type { TextOverlay, TextSource } from "./content";

/** Interface of the various `*List` types exported by the OCR lib. */
type ListLike<T> = {
  length: number;
  item(index: number): T | undefined;
};

type OCRResources = {
  detectionModel: Uint8Array;
  recognitionModel: Uint8Array;
};

let ocrResources: Promise<OCRResources> | undefined;

/**
 * Initialize the OCR engine and load models.
 *
 * This must be called before `OcrEngine*` classes can be constructed.
 */
async function initOCREngine(): Promise<OCRResources> {
  if (ocrResources) {
    return ocrResources;
  }

  const init = async () => {
    const [ocrBin, detectionModel, recognitionModel] = await Promise.all([
      fetch("../build/wasnn_ocr_bg.wasm").then((r) => r.arrayBuffer()),
      fetch("../build/text-detection.model").then((r) => r.arrayBuffer()),
      fetch("../build/text-recognition.model").then((r) => r.arrayBuffer()),
    ]);

    await initOcrLib(ocrBin);

    return {
      detectionModel: new Uint8Array(detectionModel),
      recognitionModel: new Uint8Array(recognitionModel),
    };
  };

  ocrResources = init();

  return ocrResources;
}

/**
 * Capture the visible area of the current tab.
 */
async function captureTabImage(): Promise<ImageData> {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  const activeTab = tabs[0];

  const bitmap = await chrome.tabs
    .captureVisibleTab()
    .then((dataURL) => fetch(dataURL))
    .then((response) => response.blob())
    .then((blob) =>
      createImageBitmap(blob, {
        // `captureVisibleTab` may return a HiDPI image if
        // `window.devicePixelRatio > 1`. Scaling the image down as soon as
        // possible makes subsequent operations cheaper.
        resizeWidth: activeTab.width,
        resizeHeight: activeTab.height,
      }),
    );
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext("2d")!;
  context.drawImage(bitmap, 0, 0);
  return context.getImageData(0, 0, bitmap.width, bitmap.height);
}

/**
 * Convert a list-like object returned by the OCR library into an iterator
 * that can be used with `for ... of` or `Array.from`.
 */
function* listItems<T>(list: ListLike<T>): Generator<T> {
  for (let i = 0; i < list.length; i++) {
    yield list.item(i)!;
  }
}

let recognizeText:
  | ((line: number) => Promise<LineRecResult | null>)
  | undefined;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (
    request.method === "recognizeText" &&
    typeof recognizeText === "function"
  ) {
    // TODO - Handle errors here in case `lineIndex` is invalid.
    recognizeText(request.args.lineIndex).then((result) =>
      sendResponse(result),
    );
    return true;
  }
  return false;
});

// Functions called as content scripts by the background service worker.
// These are not able to reference any external variables, except for globals
// (eg. document, window) in the environment where they run.

async function dismissTextOverlay() {
  const contentSrc = chrome.runtime.getURL("build-extension/content.js");
  const content: typeof contentModule = await import(contentSrc);
  content.dismissTextOverlay();
}

async function createTextOverlay(lines: RotatedRect[]) {
  const contentSrc = chrome.runtime.getURL("build-extension/content.js");
  const content: typeof contentModule = await import(contentSrc);
  const textSrc: TextSource = {
    recognizeText(lineIndex: number): Promise<LineRecResult | null> {
      return chrome.runtime.sendMessage({
        method: "recognizeText",
        args: { lineIndex },
      });
    },
  };
  content.createTextOverlay(textSrc, lines);
}

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.id) {
    return;
  }

  // Remove existing overlay, if extension was already activated in current tab.
  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: dismissTextOverlay,
  });

  chrome.action.setBadgeText({ text: "..." });
  try {
    const ocrInitPromise = initOCREngine();

    const captureStart = performance.now();
    const image = await captureTabImage();
    const captureEnd = performance.now();

    const { detectionModel, recognitionModel } = await ocrInitPromise;

    const ocrInit = new OcrEngineInit();
    ocrInit.setDetectionModel(detectionModel);
    ocrInit.setRecognitionModel(recognitionModel);
    const ocrEngine = new OcrEngine(ocrInit);
    const ocrInput = ocrEngine.loadImage(
      image.width,
      image.height,
      // Cast from `Uint8ClampedArray` to `Uint8Array`.
      image.data as unknown as Uint8Array,
    );

    const detStart = performance.now();
    const lines = ocrEngine.detectText(ocrInput);
    const detEnd = performance.now();

    const lineCoords = Array.from(listItems(lines)).map((line) =>
      Array.from(line.rotatedRect().corners()),
    );

    console.log(
      `Detected ${lineCoords.length} lines. Capture ${
        captureEnd - captureStart
      } ms, detection ${detEnd - detStart}ms.`,
    );

    // Set up the callback that the text layer will use to perform recognition.
    recognizeText = async (lineIndex) => {
      const line = lines.item(lineIndex);
      if (!line) {
        throw new Error("Invalid line number");
      }

      const coords = Array.from(line.rotatedRect().corners());
      const recInput = new DetectedLineList();
      recInput.push(line);

      const recLines = ocrEngine.recognizeText(ocrInput, recInput);
      const recLine = recLines.item(0);
      if (!recLine) {
        return null;
      }
      const words = Array.from(listItems(recLine.words()));

      return {
        words: words.map((word) => ({
          text: word.text(),
          coords: Array.from(word.rotatedRect().corners()),
        })),
        coords,
      };
    };

    // Create the text layer in the current tab.
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: createTextOverlay,
      args: [lineCoords],
    });
  } finally {
    chrome.action.setBadgeText({ text: "" });
  }
});
