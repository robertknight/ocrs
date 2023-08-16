/**
 * Array of coordinates of corners of a rotated rect, in the order
 * [x0, y0, x1, y1, x2, y2, x3, y3].
 */
export type RotatedRect = number[];

export type WordRecResult = {
  text: string;
  coords: RotatedRect;
};

/**
 * Serializable version of the text line recognition result (`TextLine`) from
 * the OCR engine.
 */
export type LineRecResult = {
  words: WordRecResult[];
  coords: RotatedRect;
};
