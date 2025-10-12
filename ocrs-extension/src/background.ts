import {
  DetectedLine,
  OcrEngine,
  OcrEngineInit,
  TextLine,
  default as initOcrLib,
} from "../build/ocrs.js";
import type { LineRecResult, RotatedRect, WordRecResult } from "./types";
import type * as contentModule from "./content";
import type { TextOverlay, TextOverlayOptions } from "./content";

type OCRResources = {
  detectionModel: Uint8Array;
  recognitionModel: Uint8Array;
};

let ocrResources: Promise<OCRResources> | undefined;

/**
 * Create an OCR engine and configure its models.
 */
async function createOCREngine(): Promise<OcrEngine> {
  if (!ocrResources) {
    // Initialize OCR library and fetch models on first use.
    const init = async () => {
      const [ocrBin, detectionModel, recognitionModel] = await Promise.all([
        fetch("../build/ocrs_bg.wasm").then((r) => r.arrayBuffer()),
        fetch("../build/text-detection.rten").then((r) => r.arrayBuffer()),
        fetch("../build/text-recognition.rten").then((r) => r.arrayBuffer()),
      ]);

      await initOcrLib(ocrBin);

      return {
        detectionModel: new Uint8Array(detectionModel),
        recognitionModel: new Uint8Array(recognitionModel),
      };
    };
    ocrResources = init();
  }

  const { detectionModel, recognitionModel } = await ocrResources;
  const ocrInit = new OcrEngineInit();
  ocrInit.setDetectionModel(detectionModel);
  ocrInit.setRecognitionModel(recognitionModel);
  return new OcrEngine(ocrInit);
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
 * Content script function that tests if the tab is displaying Chrome's native
 * PDF viewer.
 */
function tabIsPDFViewer() {
  return document.querySelector('embed[type="application/pdf"]') !== null;
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
async function createTextOverlay(lineCoords: RotatedRect[]) {
  const contentSrc = chrome.runtime.getURL("build-extension/content.js");
  const content: typeof contentModule = await import(contentSrc);
  const overlayOptions: TextOverlayOptions = {
    lineCoords,

    // Perform on-demand recognition of a text line.
    recognizeText(lineIndex: number): Promise<LineRecResult | null> {
      return chrome.runtime.sendMessage({
        method: "recognizeText",
        args: { lineIndex },
        time: Date.now(),
      });
    },
  };
  const overlay = content.createTextOverlay(overlayOptions);

  // Receive eagerly recognized text from the OCR engine.
  const onMessage = (message: any, sender: any) => {
    if (message.type === "textRecognized") {
      overlay.setLineText(message.lineIndex, message.recResult);
    }
  };
  chrome.runtime.onMessage.addListener(onMessage);

  // Handle removal of overlay.
  overlay.dismissed.addEventListener("abort", () => {
    chrome.runtime.onMessage.removeListener(onMessage);
    chrome.runtime.sendMessage({
      method: "cancelRecognition",
      time: Date.now(),
    });
  });
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
 * that can be serialized and sent to the tab.
 *
 * @param zoom - The tab's zoom level, as returned by `chrome.tabs.getZoom`
 * @param coords - Coordinates of the text lines from text detection
 */
function textLineToLineRecResult(
  lines: TextLine[],
  zoom: number,
  coords: RotatedRect[],
): Array<LineRecResult | null> {
  const result: Array<LineRecResult | null> = [];
  for (let i = 0; i < lines.length; i++) {
    const recLine = lines[i];
    const words = recLine.words();
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

/**
 * Map of tab ID to controller for canceling background OCR, in tabs where this
 * is currently active.
 */
const cancelControllers = new Map<number, AbortController>();

/**
 * Callback for recognizing lines on-demand in tabs where extension has most
 * recently been activated.
 *
 * TODO - There should be one of these per tab.
 */
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
  if (
    request.method === "cancelRecognition" &&
    typeof sender.tab?.id === "number"
  ) {
    const tabId = sender.tab.id;
    const cancelCtrl = cancelControllers.get(tabId);
    if (cancelCtrl) {
      cancelCtrl.abort();
      cancelControllers.delete(tabId);
    }
    return true;
  }
  return false;
});

chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.id) {
    return;
  }

  // True if the tab is using Chrome's native PDF viewer.
  let isPDF = false;

  // The tab in which the screenshot and text overlay are displayed, if we can't
  // show the overlay directly in the tab where the extension was activated.
  let screenshotTab: chrome.tabs.Tab | null = null;

  // The tab screenshot, if captured as part of creating `screenshotTab`.
  let tabImage: ImageData | null = null;

  // Determine whether this tab has content that needs special handling.
  try {
    const [isPDFResult] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: tabIsPDFViewer,
    });
    isPDF = isPDFResult.result;
  } catch (err) {
    // If the script failed to run, this may be because:
    //
    // 1. It is a tab which does not allow content scripts, even with `activeTab`
    //    permissions (eg. `chrome://` URLs).
    // 2. The page is using a sandbox origin (see
    //    https://bugs.chromium.org/p/chromium/issues/detail?id=1407986)
    //
    // Try to capture a screenshot instead and open that in a new tab with an
    // overlay.
    chrome.action.setBadgeText({ text: "..." });
    try {
      // TODO - Capture a HiDPI image here since we're going to show this
      // directly to the user.
      tabImage = await captureTabImage();
      const screenshotTabURL = new URL("/screenshot.html", location.href);
      screenshotTabURL.searchParams.set("url", tab.url ?? "");
      screenshotTab = await chrome.tabs.create({
        openerTabId: tab.id,
        url: screenshotTabURL.href,
      });
      chrome.tabs.sendMessage(screenshotTab.id!, {
        type: "imageLoaded",
        image: {
          width: tabImage.width,
          height: tabImage.height,
          data: Array.from(tabImage.data),
        },
      });
    } catch (err) {
      // We can't even capture a screenshot :( - Just show an error.
      const errorTabURL = new URL("/error.html", location.href);
      errorTabURL.searchParams.set("url", tab.url ?? "");
      chrome.tabs.create({
        openerTabId: tab.id,
        url: errorTabURL.href,
      });
      console.warn("Unable to capture tab image:", err);
      return;
    } finally {
      chrome.action.setBadgeText({ text: "" });
    }
  }

  // Remove existing overlay, if extension was already activated in current tab.
  if (!screenshotTab) {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: dismissTextOverlay,
    });
  }

  // Cancel background text recognition for existing overlay.
  let cancelCtrl = cancelControllers.get(tab.id);
  if (cancelCtrl) {
    cancelCtrl.abort();
    cancelControllers.delete(tab.id);
  }

  cancelCtrl = new AbortController();
  cancelControllers.set(tab.id, cancelCtrl);
  cancelCtrl.signal.addEventListener("abort", () => {
    chrome.action.setBadgeText({ text: "" });
  });

  chrome.action.setBadgeText({ text: "..." });

  // Get the zoom level of the tab. Tab image coordinates need to be scaled by
  // 1/zoom to map them to document coordinates. Chrome's PDF viewer is a
  // special case because the zoom level applies to the embedded native viewer,
  // but not the HTML document which contains it.
  const zoom = isPDF || screenshotTab ? 1 : await chrome.tabs.getZoom(tab.id);

  // Cache of line number to recognition result for the current image.
  const recognizedLines = new Map<number, LineRecResult | null>();

  // List of all text lines detected in the current image.
  let lines: DetectedLine[];

  try {
    // Init OCR engine concurrently with capturing tab image.
    const ocrEnginePromise = createOCREngine();

    const captureStart = performance.now();
    const image = tabImage ?? (await captureTabImage());
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
      const recInput: DetectedLine[] = [];
      const lineCoords: RotatedRect[] = [];

      for (const lineIndex of lineIndexes) {
        const line = lines[lineIndex];
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
    // line can be up to ~45% lower when recognizing a batch of similar-length
    // lines.
    recognizeText = (lineIndexes: number[]) => {
      const unrecognizedLines = lineIndexes.filter(
        (idx) => !recognizedLines.has(idx),
      );
      recognizeLineBatch(unrecognizedLines);
      return lineIndexes.map((li) => recognizedLines.get(li)!);
    };

    // Create the text layer in the current tab.
    const lineCoords = lines.map((line) =>
      tabImageToDocumentCoords(Array.from(line.rotatedRect().corners()), zoom),
    );

    if (screenshotTab) {
      chrome.tabs.sendMessage(screenshotTab.id!, {
        type: "createTextOverlay",
        lineCoords,
      });
    } else {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: createTextOverlay,
        args: [lineCoords],
      });
    }
  } finally {
    if (!cancelCtrl.signal.aborted) {
      chrome.action.setBadgeText({ text: "" });
    }
  }

  // Eagerly recognize lines, before the text layer has requested them.
  if (lines && !cancelCtrl.signal.aborted) {
    chrome.action.setBadgeText({ text: "~" });

    // Sort lines by width, so lines in each batch are likely to have similar
    // widths. This optimizes the benefit of performing recognition on a batch
    // of lines at once.
    const lineWidth = (dl: DetectedLine) => {
      const [left, top, right, bottom] = dl.rotatedRect().boundingRect();
      return right - left;
    };
    const sortedLines = [...lines];
    sortedLines.sort((a, b) => lineWidth(a) - lineWidth(b));

    // Recognize lines in batches.
    const chunkSize = 4;
    for (let indices of chunks(range(0, sortedLines.length), chunkSize)) {
      if (cancelCtrl.signal.aborted) {
        break;
      }

      // Skip any lines that were already recognized in response to a request
      // from the tab.
      indices = indices.filter((idx) => !recognizedLines.has(idx));

      const recResults = await recognizeText(indices);
      if (cancelCtrl.signal.aborted) {
        break;
      }

      for (let i = 0; i < indices.length; i++) {
        const targetTab = screenshotTab?.id ?? tab.id;
        chrome.tabs.sendMessage(targetTab, {
          type: "textRecognized",
          lineIndex: indices[i],
          recResult: recResults[i],
        });
      }

      // Pause between batches to allow any messages sent from content scripts
      // in the interim to be processed first.
      await delay(0);
    }

    if (!cancelCtrl.signal.aborted) {
      chrome.action.setBadgeText({ text: "" });
    }
  }
});
