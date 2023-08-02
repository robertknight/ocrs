import {
  DetectedLineList,
  OcrEngine,
  OcrEngineInit,
  default as initOcrLib,
} from "../build/wasnn_ocr.js";

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
async function initOCREngine() {
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

/**
 * Array of coordinates of corners of a rotated rect, in the order
 * [x0, y0, x1, y1, x2, y2, x3, y3].
 */
type RotatedRect = number[];

/**
 * Show detected text in the current tab and enable the user to select lines.
 */
function showDetectedText(lines: RotatedRect[]) {
  const canvas = document.createElement("canvas");
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  Object.assign(canvas.style, {
    position: "fixed",
    top: "0",
    left: "0",
    right: "0",
    bottom: "0",
    zIndex: 9999,
  });
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgb(0 0 0 / .3)";
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);
  document.body.append(canvas);

  // Make line polygons transparent.
  ctx.globalCompositeOperation = "destination-out";
  ctx.fillStyle = "white";

  // Map of line index to recognized text.
  const textCache = new Map();

  const linePaths = lines.map((line) => {
    const [x0, y0, x1, y1, x2, y2, x3, y3] = line;
    const path = new Path2D();
    path.moveTo(x0, y0);
    path.lineTo(x1, y1);
    path.lineTo(x2, y2);
    path.lineTo(x3, y3);
    path.closePath();
    return path;
  });

  for (const path of linePaths) {
    ctx.fill(path);
  }

  canvas.onclick = () => {
    canvas.remove();
  };

  let prevLineIndex = -1;
  canvas.onmousemove = async (e) => {
    const lineIndex = linePaths.findIndex((lp) =>
      ctx.isPointInPath(lp, e.clientX, e.clientY),
    );
    if (lineIndex === prevLineIndex) {
      return;
    }
    prevLineIndex = lineIndex;

    let text = textCache.get(lineIndex);
    if (text === undefined) {
      const recResult = await chrome.runtime.sendMessage({
        method: "recognizeText",
        args: {
          lineIndex,
        },
      });
      text = recResult.text;
      textCache.set(lineIndex, text);
    }

    console.log("Hovered line", text);
  };
}

let recognizeText: ((line: number) => Promise<string>) | undefined;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (
    request.method === "recognizeText" &&
    typeof recognizeText === "function"
  ) {
    // TODO - Handle errors here in case `lineIndex` is invalid.
    recognizeText(request.args.lineIndex).then((text) =>
      sendResponse({ text }),
    );
    return true;
  }
  return false;
});

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.id) {
    return;
  }

  // TODO - Dismiss existing captured text in current tab.

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

    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: showDetectedText,
      args: [lineCoords],
    });

    // TBD: How long does the Service Worker keep running? Will Chrome terminate
    // it while recognition requests might still come from the page?
    recognizeText = async (lineIndex) => {
      const line = lines.item(lineIndex);
      if (!line) {
        throw new Error("Invalid line number");
      }

      const recLines = new DetectedLineList();
      recLines.push(line);
      const recResult = ocrEngine.recognizeText(ocrInput, recLines);
      return recResult.item(0)!.text();
    };
  } finally {
    chrome.action.setBadgeText({ text: "" });
  }
});
