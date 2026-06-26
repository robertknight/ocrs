/*
 * C ABI for the ocrs OCR engine.
 *
 * See README.md.
 */

#ifndef OCRS_H
#define OCRS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque OCR engine handle. */
typedef struct OcrsEngine OcrsEngine;

/* A recognized line of text and the axis-aligned bounding box (in input-image
 * pixels) of its characters. `text` is owned by the parent OcrsTextLines and
 * must not be freed separately. */
typedef struct OcrsTextLine {
  char *text;
  float left;
  float top;
  float right;
  float bottom;
} OcrsTextLine;

/* Array of recognized lines. Free with `ocrs_text_lines_free`. */
typedef struct OcrsTextLines {
  OcrsTextLine *lines;
  size_t len;
} OcrsTextLines;

/* Most recent error message for the calling thread, or NULL if none. */
const char *ocrs_last_error(void);

/* ocrs version string. */
const char *ocrs_version(void);

/* Create an engine, loading models from `.rten` (or `.onnx`) files.
 * Either path may be NULL to omit that model.
 * Returns NULL on failure (see ocrs_last_error). */
OcrsEngine *ocrs_engine_new(const char *detection_model_path,
                            const char *recognition_model_path);

/* Create an engine from models held in memory. A NULL buffer pointer omits
 * that model. Buffers are only read during the call.
 * Returns NULL on failure (see ocrs_last_error). */
OcrsEngine *ocrs_engine_new_from_memory(const uint8_t *detection_model,
                                        size_t detection_model_len,
                                        const uint8_t *recognition_model,
                                        size_t recognition_model_len);

/* Release an engine. NULL is a no-op. */
void ocrs_engine_free(OcrsEngine *engine);

/* Detect and recognize all text, returned as one NUL-terminated UTF-8 string
 * with lines separated by '\n'. Requires detection and recognition models.
 * Returns NULL on failure. Free with `ocrs_string_free`. */
char *ocrs_engine_get_text(const OcrsEngine *engine, const uint8_t *image_data,
                           uint32_t width, uint32_t height, uint32_t channels);

/* Detect and recognize text as per-line text plus bounding boxes. Lines with
 * no recognized characters are omitted.
 * Returns NULL on failure. Free with `ocrs_text_lines_free`. */
OcrsTextLines *ocrs_engine_get_text_lines(const OcrsEngine *engine,
                                          const uint8_t *image_data,
                                          uint32_t width, uint32_t height,
                                          uint32_t channels);

/* Free a string from `ocrs_engine_get_text`. NULL is a no-op. */
void ocrs_string_free(char *text);

/* Free a result from `ocrs_engine_get_text_lines`, including line text.
 * NULL is a no-op. */
void ocrs_text_lines_free(OcrsTextLines *lines);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OCRS_H */
