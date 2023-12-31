import { createTextOverlay } from "./content.js";
import type { TextOverlay } from "./content";
import type { LineRecResult } from "./types";

const screenshotCanvas = document.getElementById(
  "screenshotImage",
) as HTMLCanvasElement;

const screenshotContainer = document.getElementById("container")!;

let overlay: TextOverlay | null = null;

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.type) {
    case "createTextOverlay":
      overlay = createTextOverlay({
        autoDismiss: false,
        container: screenshotContainer,
        lineCoords: message.lineCoords,
        recognizeText(lineIndex: number): Promise<LineRecResult | null> {
          return chrome.runtime.sendMessage({
            method: "recognizeText",
            args: { lineIndex },
            time: Date.now(),
          });
        },
      });
      break;
    case "imageLoaded":
      const { width, height, data } = message.image;
      screenshotCanvas.width = width;
      screenshotCanvas.height = height;
      const imgData = new ImageData(new Uint8ClampedArray(data), width, height);
      const ctx = screenshotCanvas.getContext("2d")!;
      ctx.putImageData(imgData, 0, 0);
      break;
    case "textRecognized":
      overlay?.setLineText(message.lineIndex, message.recResult);
      break;
  }
});

const params = new URLSearchParams(location.search);
const sourceURIField = document.getElementById("sourceTabURI")!;
const url = params.get("url");
sourceURIField.textContent = url ?? "(unknown URL)";

// Tell TS this is an ES module.
export {};
