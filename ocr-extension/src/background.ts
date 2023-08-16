import {
  DetectedLine,
  DetectedLineList,
  OcrEngine,
  OcrEngineInit,
  TextLineList,
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
  | ((lineIndexes: number[]) => Array<LineRecResult | null>)
  | undefined;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (
    request.method === "recognizeText" &&
    typeof recognizeText === "function"
  ) {
    const [result] = recognizeText([request.args.lineIndex]);
    sendResponse(result);
    return true;
  }
  return false;
});

function* chunks<T>(items: T[], chunkSize: number) {
  for (let i = 0; i < items.length; i += chunkSize) {
    yield items.slice(i, i + chunkSize);
  }
}

/** Return an array of numbers in the range `[start, end)` */
function range(start: number, end: number) {
  return Array(end - start)
    .fill(start)
    .map((x, i) => x + i);
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Content script function that removes the overlay in the current tab.
 */
async function dismissTextOverlay() {
  const contentSrc = chrome.runtime.getURL("build-extension/content.js");
  const content: typeof contentModule = await import(contentSrc);
  content.dismissTextOverlay();
}

/**
 * Content script function that creates an overlay in the current tab.
 */
async function createTextOverlay(lines: RotatedRect[]) {
  const contentSrc = chrome.runtime.getURL("build-extension/content.js");
  const content: typeof contentModule = await import(contentSrc);
  const textSrc: TextSource = {
    // Perform on-demand recognition of a text line.
    recognizeText(lineIndex: number): Promise<LineRecResult | null> {
      return chrome.runtime.sendMessage({
        method: "recognizeText",
        args: { lineIndex },
        time: Date.now(),
      });
    },
  };
  const overlay = content.createTextOverlay(textSrc, lines);

  // Receive eagerly recognized text from the OCR engine.
  const onMessage = (message: any, sender: any) => {
    if (message.type === "textRecognized") {
      overlay.setLineText(message.lineIndex, message.recResult);
    }
  };
  chrome.runtime.onMessage.addListener(onMessage);
  overlay.dismissed.addEventListener("abort", () => {
    chrome.runtime.onMessage.removeListener(onMessage);
  });
}

async function createOCREngine() {
  const { detectionModel, recognitionModel } = await initOCREngine();
  const ocrInit = new OcrEngineInit();
  ocrInit.setDetectionModel(detectionModel);
  ocrInit.setRecognitionModel(recognitionModel);
  return new OcrEngine(ocrInit);
}

/**
 * Map coordinates from a tab screenshot to the coordinate system of the
 * document's viewport, as seen by code running in the tab.
 *
 * This assumes that the screenshot was captured at regular DPI (ie. as if
 * `window.devicePixelRatio` was 1), and so we don't need to compensate for
 * that here.
 */
function tabImageToDocumentCoords(coords: number[], zoom: number) {
  return coords.map((c) => c / zoom);
}

/**
 * Convert a `TextLineList` from the OCR engine to a `LineRecResult` array
 * that can be serialized and send to the tab.
 */
function textLineToLineRecResult(
  lines: TextLineList,
  zoom: number,
  coords: RotatedRect[],
): Array<LineRecResult | null> {
  const result: Array<LineRecResult | null> = [];
  for (let i = 0; i < lines.length; i++) {
    const recLine = lines.item(i);
    if (!recLine) {
      result.push(null);
      continue;
    }
    const words = Array.from(listItems(recLine.words()));
    result.push({
      words: words.map((word) => ({
        text: word.text(),
        coords: tabImageToDocumentCoords(
          Array.from(word.rotatedRect().corners()),
          zoom,
        ),
      })),
      coords: tabImageToDocumentCoords(coords[i], zoom),
    });
  }
  return result;
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

  const zoom = await chrome.tabs.getZoom(tab.id);

  // Cache of line number to recognition result for the current image.
  const recognizedLines = new Map<number, LineRecResult | null>();

  // List of all text lines detected in the current image.
  let lines: DetectedLineList;

  try {
    const ocrEnginePromise = createOCREngine();

    const captureStart = performance.now();
    const image = await captureTabImage();
    const captureEnd = performance.now();

    const ocrEngine = await ocrEnginePromise;
    const ocrInput = ocrEngine.loadImage(
      image.width,
      image.height,
      // Cast from `Uint8ClampedArray` to `Uint8Array`.
      image.data as unknown as Uint8Array,
    );

    const detStart = performance.now();
    lines = ocrEngine.detectText(ocrInput);
    const detEnd = performance.now();

    console.log(
      `Detected ${lines.length} lines. Capture ${
        captureEnd - captureStart
      } ms, detection ${detEnd - detStart}ms.`,
    );

    /**
     * Perform recognition on a batch of lines and cache the results.
     */
    const recognizeLineBatch = (lineIndexes: number[]) => {
      const recInput = new DetectedLineList();
      const lineCoords: RotatedRect[] = [];

      for (const lineIndex of lineIndexes) {
        const line = lines.item(lineIndex);
        if (!line) {
          throw new Error("Invalid line number");
        }
        lineCoords.push(Array.from(line.rotatedRect().corners()));
        recInput.push(line);
      }

      const textLines = ocrEngine.recognizeText(ocrInput, recInput);
      for (const [i, recLine] of textLineToLineRecResult(
        textLines,
        zoom,
        lineCoords,
      ).entries()) {
        recognizedLines.set(lineIndexes[i], recLine);
      }
    };

    // Set up callback that performs text recognition.
    //
    // This takes a list of line indexes as input. The recognition time per
    // line can be up to ~45% lower when recognizing a batch of similiar-length
    // lines.
    recognizeText = (lineIndexes: number[]) => {
      const unrecognizedLines = lineIndexes.filter(
        (idx) => !recognizedLines.has(idx),
      );
      recognizeLineBatch(unrecognizedLines);
      return lineIndexes.map((li) => recognizedLines.get(li)!);
    };

    // Create the text layer in the current tab.
    const lineCoords = Array.from(listItems(lines)).map((line) =>
      tabImageToDocumentCoords(Array.from(line.rotatedRect().corners()), zoom),
    );

    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: createTextOverlay,
      args: [lineCoords],
    });
  } finally {
    chrome.action.setBadgeText({ text: "" });
  }

  // Eagerly recognize lines, before the text layer has requested them.
  if (lines) {
    chrome.action.setBadgeText({ text: "~" });

    // Sort lines by width, so lines in each batch are likely to have similar
    // widths. This optimizes the benefit of performing recognition on a batch
    // of lines at once.
    const lineWidth = (dl: DetectedLine) => {
      const [left, top, right, bottom] = dl.rotatedRect().boundingRect();
      return right - left;
    };
    const sortedLines = [...listItems(lines)];
    sortedLines.sort((a, b) => lineWidth(a) - lineWidth(b));

    // Recognize lines in batches.
    const chunkSize = 4;
    for (let indices of chunks(range(0, sortedLines.length), chunkSize)) {
      // Skip any lines that were already recognized in response to a request
      // from the tab.
      indices = indices.filter((idx) => !recognizedLines.has(idx));

      const recResults = await recognizeText(indices);
      for (let i = 0; i < indices.length; i++) {
        chrome.tabs.sendMessage(tab.id, {
          type: "textRecognized",
          lineIndex: indices[i],
          recResult: recResults[i],
        });
      }

      // Pause between batches to allow any messages sent from content scripts
      // in the interim to be processed first.
      await delay(0);
    }

    chrome.action.setBadgeText({ text: "" });
  }
});
