import type { LineRecResult, RotatedRect, WordRecResult } from "./types";

/**
 * See https://developer.mozilla.org/en-US/docs/Web/API/CaretPosition
 */
type TextPosition = { textNode: Text; offset: number };

/**
 * Return true if the point `(x, y)` is contained within `r`.
 *
 * The left/top edges are treated as "inside" the rect and the bottom/right
 * edges as "outside". This ensures that for adjacent rects, a point will only
 * lie within one rect.
 */
function rectContains(r: DOMRect, x: number, y: number) {
  return x >= r.left && x < r.right && y >= r.top && y < r.bottom;
}

/**
 * Return the offset of the text character which in a {@link Text} node that is
 * a descendant of `container`, and contains the point `(x, y)`, specified in
 * client coordinates.
 */
function textPositionFromPoint(
  container: Element,
  x: number,
  y: number,
): TextPosition | null {
  // TODO - Optimize this using `{Document, ShadowRoot}.elementsFromPoint` to
  // filter the node list.
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  let currentNode;
  const range = new Range();

  while ((currentNode = walker.nextNode())) {
    const text = currentNode as Text;
    const str = text.nodeValue!;
    for (let i = 0; i < str.length; i++) {
      range.setStart(text, i);
      range.setEnd(text, i + 1);
      const charRect = range.getBoundingClientRect();
      if (rectContains(charRect, x, y)) {
        return { textNode: text, offset: i };
      }
    }
  }
  return null;
}

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

    // Avoid line break if word elements don't quite fit. Also preserve spaces
    // at the end of the line.
    whiteSpace: "pre",

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

  const spaceWidth = context.measureText(" ").width;

  // Add words to the line as inline-block elements. This allows us to create
  // normal text selection behavior while adjusting the positioning and size
  // of words to match the underlying pixels.
  let prevWordRect: DOMRect | undefined;
  let prevWordEl: HTMLElement | undefined;
  for (const word of line.words) {
    const wordRect = domRectFromRotatedRect(word.coords);
    const leftMargin = prevWordRect
      ? wordRect.left - prevWordRect.right - spaceWidth
      : undefined;

    // Create outer element for word. This sets the width and margin used for
    // inline layout.
    const wordEl = document.createElement("span");
    Object.assign(wordEl.style, {
      display: "inline-block",
      marginLeft: leftMargin != null ? `${leftMargin}px` : undefined,
      width: `${wordRect.width}px`,
      height: `${wordRect.height}px`,
    });

    const metrics = context.measureText(word.text);
    const xScale = wordRect.width / metrics.width;
    const yScale = wordRect.height / fixedFont.size;

    // Create inner element for word. This uses a transform to make the rendered
    // size match the underlying text pixels. The transform doesn't affect
    // layout. The inner and outer elements are separated so that the scale
    // transform is not applied to the hit box for the outer element, as that
    // would make the hit box's size `(width * xScale, height * yScale)`, which
    // can interfere with selection of subsequent words.
    const wordInner = document.createElement("span");
    wordInner.textContent = word.text;
    Object.assign(wordInner.style, {
      display: "inline-block",
      transformOrigin: "top left",
      transform: `scale(${xScale}, ${yScale})`,
    });
    wordEl.append(wordInner);

    lineEl.append(wordEl);
    prevWordEl = wordEl;
    prevWordRect = wordRect;

    // Add space between words. We add this even after the last word in a line
    // to ensure there is a space between the end of one line and the start of
    // the next in a multi-line selection.
    //
    // TODO - Adjust the rendered size of the space to make the height of
    // selection boxes consistent across the line, and avoid gaps in the
    // selection box.
    lineEl.append(" ");
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

    // Position the overlay so that it fills the viewport, but scrolls with
    // the page contents. This allows the user to read parts of the page that
    // were OCR-ed, without disrupting the selection in the part that has been.
    //
    // A known issue with this is that text in the overlay that belongs to
    // fixed-positioned elements won't scroll as the page moves.
    position: "absolute",
    top: `${document.documentElement.scrollTop}px`,
    left: `${document.documentElement.scrollLeft}px`,
    width: `${window.innerWidth}px`,
    height: `${window.innerHeight}px`,
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

  canvasContainer.shadowRoot!.append(canvas);

  // Draw text layer backdrop.
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgb(0 0 0 / .3)";
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

  // Make line polygons transparent.
  ctx.globalCompositeOperation = "destination-out";
  ctx.fillStyle = "white";

  // Map of line index to:
  //  1) Recognized text, if recognition is complete
  //  2) A promise if recognition is in progress
  //  3) `null` if recognition completed but no text was recognized
  const textCache = new Map<
    number,
    LineRecResult | Promise<LineRecResult | null> | null
  >();

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

  const lineIndexFromPoint = (clientX: number, clientY: number) => {
    const canvasRect = canvas.getBoundingClientRect();
    const canvasX = clientX - canvasRect.left;
    const canvasY = clientY - canvasRect.top;
    return linePaths.findIndex((lp) => ctx.isPointInPath(lp, canvasX, canvasY));
  };

  // Track the start and end coordinates of the current mouse drag operation.
  const primaryButtonPressed = (e: MouseEvent) => e.buttons & 1;
  let dragStartAt: { x: number; y: number } | null = null;
  let dragEndAt: { x: number; y: number } | null = null;
  canvasContainer.onmousedown = (e) => {
    if (!dragStartAt) {
      dragStartAt = { x: e.x, y: e.y };
    }
  };
  canvasContainer.onmousemove = (e) => {
    if (primaryButtonPressed(e)) {
      dragEndAt = { x: e.x, y: e.y };
    }
  };
  canvasContainer.onmouseup = (e) => {
    dragEndAt = null;
    dragStartAt = null;
  };
  canvasContainer.onmouseenter = (e) => {
    if (primaryButtonPressed(e)) {
      dragStartAt = { x: e.x, y: e.y };
    } else {
      dragStartAt = null;
    }
  };

  const recognizeLine = async (
    lineIndex: number,
  ): Promise<LineRecResult | null> => {
    const cachedResult = textCache.get(lineIndex);
    if (cachedResult !== undefined) {
      return cachedResult;
    }

    const recPromise: Promise<LineRecResult | null> =
      chrome.runtime.sendMessage({
        method: "recognizeText",
        args: {
          lineIndex,
        },
      });
    textCache.set(lineIndex, recPromise);

    const recResult = await recPromise;
    textCache.set(lineIndex, recResult);

    if (!recResult) {
      return recResult;
    }

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

    // If a drag operation was in progress, update the selection once text
    // recognition is completed, to include the newly recognized text.
    if (dragStartAt && dragEndAt) {
      const start = textPositionFromPoint(
        textLayer,
        dragStartAt.x,
        dragStartAt.y,
      );
      const end = textPositionFromPoint(textLayer, dragEndAt.x, dragEndAt.y);
      const selection = document.getSelection();
      if (selection && start && end) {
        selection.setBaseAndExtent(
          start.textNode,
          start.offset,
          end.textNode,
          end.offset,
        );
      }
    }

    return recResult;
  };

  // Create the hidden text layer in which the user can select text.
  const textLayer = document.createElement("div");
  canvasContainer.shadowRoot!.append(textLayer);

  const textLines = new Map<number, HTMLElement>();

  let prevLine = -1;
  canvas.onmousemove = async (e) => {
    const currentLine = lineIndexFromPoint(e.clientX, e.clientY);
    if (currentLine === prevLine) {
      return;
    }
    prevLine = currentLine;

    if (currentLine === -1) {
      return;
    }

    // Recognize text for the current line, if just hovering, or all lines
    // between the start and end point of the current drag operation if a mouse
    // button is pressed.
    const dragStartLine = dragStartAt
      ? lineIndexFromPoint(dragStartAt.x, dragStartAt.y)
      : currentLine;
    const startLine =
      dragStartLine !== -1 ? Math.min(dragStartLine, currentLine) : currentLine;
    const endLine =
      dragStartLine !== -1 ? Math.max(dragStartLine, currentLine) : currentLine;

    for (let lineIndex = startLine; lineIndex <= endLine; lineIndex++) {
      if (!textCache.has(lineIndex)) {
        // TODO: We currently recognize a single line at a time here, but
        // recognition is more efficient if batches of similarly-sized lines
        // are recognized at once.
        recognizeLine(lineIndex);
      }
    }
  };

  const removeOverlay = () => canvasContainer.remove();

  // Dismiss overlay when user clicks on the backdrop, but not inside text or
  // other UI elements in the overlay.
  canvas.onclick = (e) => {
    // Don't dismiss the overlay if the user started a drag action (eg. to
    // select text), but happened to finish on the canvas instead of inside a
    // text element.
    if (dragStartAt) {
      const dragDist = Math.sqrt(
        (e.x - dragStartAt.x) ** 2 + (e.y - dragStartAt.y) ** 2,
      );
      if (dragDist >= 20) {
        return;
      }
    }

    // Don't dismiss the overlay if the user clicks inside a line that hasn't
    // been recognized yet.
    const lineIndex = lineIndexFromPoint(e.clientX, e.clientY);
    if (lineIndex !== -1) {
      return;
    }

    removeOverlay();
  };

  // When the window is resized, the document layout will change and the OCR
  // boxes will likely be incorrect, so just remove the overlay at that point.
  window.addEventListener(
    "resize",
    () => {
      removeOverlay();
    },
    { once: true },
  );
}
