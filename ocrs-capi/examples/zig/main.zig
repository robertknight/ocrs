//! Minimal Zig example
//!
//! This is intentionally self-contained: the model paths, image dimensions, and
//! raw pixel data are hardcoded. Makes the example simpler instead of going
//! through Zig Io for the sake of the example.
//!
//! See `README.md` for how to produce `text-detection.rten`,
//! `text-recognition.rten` and the raw `image.rgb` referenced below, and for
//! the exact `zig build-exe` command used to link against `libocrs_capi`.

const std = @import("std");
const c = @cImport({
    @cInclude("ocrs.h");
});

// Paths to the model files, relative to the working directory at run time.
const detection_model = "text-detection.rten";
const recognition_model = "text-recognition.rten";

// Raw 8-bit pixels in row-major, channels-last (HWC) order. Generate this with
// the Python snippet in README.md and update the dimensions to match.
const image_data = @embedFile("image.rgb");
const width: u32 = 2320;
const height: u32 = 776;
const channels: u32 = 3; // 1 = grey, 3 = RGB, 4 = RGBA

pub fn main() !void {
    std.debug.print("ocrs {s}\n", .{c.ocrs_version()});

    const engine = c.ocrs_engine_new(detection_model, recognition_model) orelse {
        std.debug.print("failed to create engine: {s}\n", .{c.ocrs_last_error()});
        return error.EngineCreationFailed;
    };
    defer c.ocrs_engine_free(engine);

    // Whole-image text as a single string.
    const text = c.ocrs_engine_get_text(engine, image_data, width, height, channels) orelse {
        std.debug.print("OCR failed: {s}\n", .{c.ocrs_last_error()});
        return error.OcrFailed;
    };
    defer c.ocrs_string_free(text);
    std.debug.print("--- text ---\n{s}\n", .{text});

    // Per-line text with bounding boxes.
    const lines = c.ocrs_engine_get_text_lines(engine, image_data, width, height, channels) orelse {
        std.debug.print("OCR failed: {s}\n", .{c.ocrs_last_error()});
        return error.OcrFailed;
    };
    defer c.ocrs_text_lines_free(lines);

    std.debug.print("--- {d} lines ---\n", .{lines.*.len});
    var i: usize = 0;
    while (i < lines.*.len) : (i += 1) {
        const line = lines.*.lines[i];
        std.debug.print("[{d:.0},{d:.0},{d:.0},{d:.0}] {s}\n", .{
            line.left, line.top, line.right, line.bottom, line.text,
        });
    }
}
