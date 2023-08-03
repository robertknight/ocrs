import type { LineRecResult, RotatedRect, WordRecResult } from "./types";

/**
 * Return the smallest axis-aligned rect that contains all corners of a
 * rotated rect.
 */
function domRectFromRotatedRect(coords: RotatedRect): DOMRect {
  const [x0, y0, x1, y1, x2, y2, x3, y3] = coords;
  const left = Math.min(x0, x1, x2, x3);
  const top = Math.min(y0, y1, y2, y3);
  const right = Math.max(x0, x1, x2, x3);
  const bottom = Math.max(y0, y1, y2, y3);
  return new DOMRect(left, top, right - left, bottom - top);
}

/** Font used for transparent text layer content. */
const fixedFont = {
  size: 16,
  family: "sans-serif",
};

/**
 * Create a line of selectable text in the transparent text layer.
 */
function createTextLine(line: LineRecResult) {
  const lineEl = document.createElement("div");
  lineEl.className = "text-line";

  const { left, top, right, bottom } = domRectFromRotatedRect(line.coords);
  Object.assign(lineEl.style, {
    // Position transparent line above text
    position: "absolute",
    left: `${left}px`,
    top: `${top}px`,
    width: `${right - left}px`,
    height: `${bottom - top}px`,

    // Avoid line break if word elements don't quite fit.
    whiteSpace: "nowrap",

    // Use a fixed font. This needs to match the font used when measuring the
    // natural width of text.
    fontSize: `${fixedFont.size}px`,
    fontFamily: fixedFont.family,

    // Make text transparent
    color: "rgb(0 0 0 / 0)",
  });

  // Create canvas for measuring natural size of text.
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d")!;
  context.font = `${fixedFont.size}px ${fixedFont.family}`;

  // Add words to the line as inline-block elements. This allows us to create
  // normal text selection behavior while adjusting the positioning and size
  // of words to match the underlying pixels.
  let prevWordRect: DOMRect | undefined;
  let prevWordEl: HTMLElement | undefined;
  for (const word of line.words) {
    if (prevWordEl) {
      prevWordEl.textContent += " ";
    }

    const wordRect = domRectFromRotatedRect(word.coords);
    const metrics = context.measureText(word.text);
    const xScale = wordRect.width / metrics.width;
    const yScale = wordRect.height / fixedFont.size;
    const leftMargin = prevWordRect
      ? wordRect.left - prevWordRect.right
      : undefined;

    const wordEl = document.createElement("span");
    wordEl.textContent = word.text;
    Object.assign(wordEl.style, {
      display: "inline-block",
      transformOrigin: "top left",

      marginLeft: leftMargin != null ? `${leftMargin}px` : undefined,

      // Set the size of this box used for layout. This does not affect the
      // rendered size.
      width: `${wordRect.width}px`,
      height: `${wordRect.height}px`,

      // Scale content using a transform. This affects the rendered size of the
      // text, but not the layout.
      transform: `scale(${xScale}, ${yScale})`,

      // Don't collapse whitespace at end of words, so it remains visible in
      // selected text.
      whiteSpace: "pre",
    });

    lineEl.append(wordEl);
    prevWordEl = wordEl;
    prevWordRect = wordRect;
  }
  return lineEl;
}

/**
 * Show detected text in the current tab and enable the user to select lines.
 */
export function showDetectedText(lines: RotatedRect[]) {
  const canvasContainer = document.createElement("div");
  Object.assign(canvasContainer.style, {
    // Override default styles from the page.
    all: "initial",

    position: "fixed",
    top: "0",
    left: "0",
    right: "0",
    bottom: "0",
    zIndex: 9999,
  });

  // Use a shadow root to insulate children from page styles.
  canvasContainer.attachShadow({ mode: "open" });

  document.body.append(canvasContainer);

  const canvas = document.createElement("canvas");
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  Object.assign(canvas.style, {
    position: "absolute",
    top: "0",
    left: "0",
    right: "0",
    bottom: "0",
  });

  // Dismiss overlay when user clicks on the backdrop, but not inside text or
  // other UI elements in the overlay.
  canvas.onclick = (e) => {
    canvasContainer.remove();
  };

  canvasContainer.shadowRoot!.append(canvas);

  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgb(0 0 0 / .3)";
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

  // Make line polygons transparent.
  ctx.globalCompositeOperation = "destination-out";
  ctx.fillStyle = "white";

  // Map of line index to recognized text.
  const textCache = new Map<number, LineRecResult | null>();

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

  const textLayer = document.createElement("div");
  canvasContainer.shadowRoot!.append(textLayer);

  // Map of line index to text line element.
  const textLines = new Map<number, HTMLElement>();

  let prevLineIndex = -1;
  canvas.onmousemove = async (e) => {
    const lineIndex = linePaths.findIndex((lp) =>
      ctx.isPointInPath(lp, e.clientX, e.clientY),
    );
    if (lineIndex === prevLineIndex) {
      return;
    }
    prevLineIndex = lineIndex;

    let cachedResult = textCache.get(lineIndex);
    if (cachedResult === undefined) {
      const recResult: LineRecResult = await chrome.runtime.sendMessage({
        method: "recognizeText",
        args: {
          lineIndex,
        },
      });

      const lineEl = createTextLine(recResult);
      lineEl.setAttribute("data-line-index", lineIndex.toString());

      // Insert line such that the DOM order is the same as the output order
      // from the OCR lib, which produces lines in reading order. This makes
      // text selection across lines and columns flow properly, provided that
      // the OCR lib detected the reading order correctly.
      const successor = Array.from(textLines.entries())
        .sort((a, b) => a[0] - b[0])
        .find(([entryLine, entryEl]) => entryLine >= lineIndex);
      const successorNode = successor ? successor[1] : null;
      textLayer.insertBefore(lineEl, successorNode);

      textLines.set(lineIndex, lineEl);
      textCache.set(lineIndex, recResult);
      cachedResult = recResult;
    }
  };
}
