import type { LineRecResult, RotatedRect, WordRecResult } from "./types";

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
 * Return the text node and offset of the character at the point `(x, y)` in
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
function createTextLine(line: LineRecResult): HTMLElement {
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
  for (const [wordIndex, word] of line.words.entries()) {
    const wordRect = domRectFromRotatedRect(word.coords);
    const leftMargin = prevWordRect
      ? wordRect.left - prevWordRect.right - spaceWidth
      : wordRect.left - left;

    // Create outer element for word. This sets the width and margin used for
    // inline layout.
    const wordEl = document.createElement("span");
    Object.assign(wordEl.style, {
      display: "inline-block",
      marginLeft: leftMargin != null ? `${leftMargin}px` : undefined,
      marginTop: `${wordRect.top - top}px`,
      width: `${wordRect.width}px`,
      height: `${wordRect.height}px`,

      // Align top of word box with top of line.
      verticalAlign: "top",
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
    const spaceEl = document.createElement("span");

    let spaceXScale = 1;
    if (wordIndex < line.words.length - 1) {
      const nextWordRect = domRectFromRotatedRect(
        line.words[wordIndex + 1].coords,
      );
      const targetSpaceWidth = nextWordRect.left - wordRect.right;
      spaceXScale = targetSpaceWidth / spaceWidth;
    }

    Object.assign(spaceEl.style, {
      display: "inline-block",

      // Align top of space with top of preceding word.
      marginTop: `${wordRect.top - top}px`,
      verticalAlign: "top",

      // Scale the space to match the height of the word, and the width between
      // the current and next words.
      transformOrigin: "top left",
      transform: `scale(${spaceXScale}, ${yScale})`,
    });
    spaceEl.textContent = " ";

    lineEl.append(spaceEl);
  }
  return lineEl;
}

export type TextOverlay = {
  /** A signal that fires when the overlay is removed. */
  dismissed: AbortSignal;

  /** Remove the overlay. */
  remove(): void;

  /**
   * Set the recognized text for a line, or `null` if no text is found.
   *
   * This can be used by the OCR engine to supply recognition results before
   * they have been requested as a result of eg. the user hovering a line,
   * avoiding latency when recognition is needed later on.
   */
  setLineText(lineIndex: number, recResult: LineRecResult | null): void;
};

/**
 * Interface for the text layer in a tab to communicate with the OCR engine.
 */
export type TextSource = {
  /**
   * Run recognition on a line of text and return the recognition results, or
   * `null` if no text was recognized.
   */
  recognizeText(lineIndex: number): Promise<LineRecResult | null>;
};

/** The active overlay in the current document. */
let activeOverlay: TextOverlay | null = null;

/**
 * Remove the active overlay in the current document.
 */
export function dismissTextOverlay() {
  activeOverlay?.remove();
  activeOverlay = null;
}

/**
 * Return the container element into which the text overlay should be placed.
 *
 * By default this is the document body, but if there is a top layer [1] active,
 * the overlay needs to be placed in that to be visible.
 *
 * [1] https://developer.mozilla.org/en-US/docs/Glossary/Top_layer
 */
function getOverlayParent() {
  if (document.fullscreenElement) {
    return document.fullscreenElement;
  }

  // Other ways of creating a top layer that are not yet handled:
  // - `HTMLDialogElement.showModal()`
  // - `HTMLElement.showPopover()`

  return document.body;
}

/**
 * Create an overlay which shows the location of OCR-ed text in the viewport
 * and enables the user to select and copy text from it.
 *
 * Only one overlay is supported in the document at a time, and if there is
 * already an existing active overlay when this function is called, it is
 * dismissed.
 *
 * @param source - Interface for communicating with the OCR engine
 * @param lines - Locations of text lines in the page
 */
export function createTextOverlay(
  source: TextSource,
  lines: RotatedRect[],
): TextOverlay {
  dismissTextOverlay();

  // Pending recognition requests. These are processed in LIFO order as
  // requests are triggered in response to user interactions (eg. hovering a
  // text line) and so we want to give priority to the most recently hovered
  // line.
  const pendingRecRequests: Array<{
    lineIndex: number;
    resolve: (result: LineRecResult | null) => void;
  }> = [];
  let pendingRecTimer: number | undefined;

  const flushPendingRequests = () => {
    while (pendingRecRequests.length > 0) {
      const req = pendingRecRequests.pop()!;
      source.recognizeText(req.lineIndex).then((result) => req.resolve(result));
    }
    pendingRecTimer = undefined;
  };

  // Schedule recognition of a line. We buffer requests that happen close
  // together and process them in LIFO order.
  const scheduleRecognition = (lineIndex: number) => {
    let resolve;
    const recResult = new Promise<LineRecResult | null>((resolve_) => {
      resolve = resolve_;
    });

    pendingRecRequests.push({ lineIndex, resolve: resolve! });
    clearTimeout(pendingRecTimer);
    // nb. Node typings for `setTimeout` are incorrectly being used here.
    pendingRecTimer = setTimeout(
      flushPendingRequests,
      100,
    ) as unknown as number;

    return recResult;
  };

  const canvasContainer = document.createElement("div");
  Object.assign(canvasContainer.style, {
    // Override default styles from the page.
    all: "initial",

    // Position the overlay so that it fills the viewport, but scrolls with
    // the page contents. This allows the user to read parts of the page that
    // were OCR-ed, without disrupting the selection in the part that has been.
    //
    // A known issue with this is that when the page is scrolled, text in the
    // overlay will become mis-aligned with underlying pixels that belong to
    // fixed-positioned elements.
    position: "absolute",
    top: `${document.documentElement.scrollTop}px`,
    left: `${document.documentElement.scrollLeft}px`,
    width: `${window.innerWidth}px`,
    height: `${window.innerHeight}px`,

    // Display overlay above other elements. If there is a top layer active,
    // then we also need to ensure the element is added to that layer.
    zIndex: 9999,
  });

  // Use a shadow root to insulate children from page styles.
  canvasContainer.attachShadow({ mode: "open" });

  const overlayParent = getOverlayParent();
  overlayParent.append(canvasContainer);

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

  // Map of line index to:
  //  1) Recognized text, if recognition is complete
  //  2) A promise if recognition is in progress
  //  3) `null` if recognition completed but no text was recognized
  const textCache = new Map<
    number,
    LineRecResult | Promise<LineRecResult | null> | null
  >();

  const rotatedRectPath = (coords: RotatedRect) => {
    const [x0, y0, x1, y1, x2, y2, x3, y3] = coords;
    const path = new Path2D();
    path.moveTo(x0, y0);
    path.lineTo(x1, y1);
    path.lineTo(x2, y2);
    path.lineTo(x3, y3);
    path.closePath();
    return path;
  };

  // Cut out holes in the backdrop where there is detected text.
  ctx.save();
  ctx.globalCompositeOperation = "destination-out";
  ctx.fillStyle = "white";
  const linePaths = lines.map(rotatedRectPath);
  for (const path of linePaths) {
    ctx.fill(path);
  }
  ctx.restore();

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

  /**
   * Save recognition results for a line and, if not null, create the
   * transparent text line allowing the user to select text.
   */
  const initTextLine = (lineIndex: number, recResult: LineRecResult | null) => {
    textCache.set(lineIndex, recResult);
    if (!recResult) {
      return;
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
  };

  // Perform on-demand recognition when the user hovers a line of text that has
  // not yet been recognized.
  const recognizeLine = async (
    lineIndex: number,
  ): Promise<LineRecResult | null> => {
    const cachedResult = textCache.get(lineIndex);
    if (cachedResult !== undefined) {
      return cachedResult;
    }

    const recPromise = scheduleRecognition(lineIndex);
    textCache.set(lineIndex, recPromise);

    const recResult = await recPromise;
    initTextLine(lineIndex, recResult);
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

  // Signal used to remove global listeners etc. when overlay is removed.
  const overlayRemoved = new AbortController();
  overlayRemoved.signal.addEventListener("abort", () => {
    canvasContainer.remove();
    clearTimeout(pendingRecTimer);
  });

  const removeOverlay = () => overlayRemoved.abort();

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
    { signal: overlayRemoved.signal },
  );

  document.addEventListener(
    "keyup",
    (e) => {
      if (e.key === "Escape") {
        removeOverlay();
      }
    },
    { signal: overlayRemoved.signal },
  );

  activeOverlay = {
    dismissed: overlayRemoved.signal,
    setLineText: (lineIndex: number, recResult: LineRecResult) => {
      if (textCache.has(lineIndex)) {
        return;
      }
      initTextLine(lineIndex, recResult);
    },
    remove: removeOverlay,
  };
  return activeOverlay;
}
