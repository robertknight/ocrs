//! Integration tests that run the full detect, layout, recognize pipeline
//! through the C ABI.
//!
//! Read the model paths from environment variables:
//!
//!   OCRS_DETECTION_MODEL   = path to text-detection.rten
//!   OCRS_RECOGNITION_MODEL = path to text-recognition.rten
//!
//! Run them (downloading the models first) with:
//!
//!   make test-capi
//!
//! or manually, with absolute paths:
//!
//!   OCRS_DETECTION_MODEL=... OCRS_RECOGNITION_MODEL=... \
//!     cargo test -p ocrs-capi --test ocr -- --ignored
//!
//! If the variables are unset the tests print a notice and pass without doing
//! anything, so `cargo test -- --ignored` stays green when no models available.

use std::ffi::{CStr, CString};
use std::path::Path;

use ocrs_capi::{
    ocrs_engine_free, ocrs_engine_get_text, ocrs_engine_get_text_lines, ocrs_engine_new,
    ocrs_last_error, ocrs_string_free, ocrs_text_lines_free, OcrsEngine,
};

/// Image with known text.
const TEST_IMAGE: &str = "../ocrs-cli/test-data/why-rust.png";

fn models_from_env() -> Option<(CString, CString)> {
    let det = std::env::var("OCRS_DETECTION_MODEL").ok()?;
    let rec = std::env::var("OCRS_RECOGNITION_MODEL").ok()?;
    Some((CString::new(det).unwrap(), CString::new(rec).unwrap()))
}

fn last_error() -> String {
    let ptr = ocrs_last_error();
    if ptr.is_null() {
        "<none>".into()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

/// Decode the test image to raw HWC RGB bytes plus its dimensions.
fn load_rgb(path: &str) -> (Vec<u8>, u32, u32) {
    let image = image::open(Path::new(path))
        .unwrap_or_else(|err| panic!("failed to open {path}: {err}"))
        .into_rgb8();
    let (width, height) = image.dimensions();
    (image.into_raw(), width, height)
}

/// Build an engine from the env-provided models, or `None` to skip the test.
fn engine_or_skip(test: &str) -> Option<*mut OcrsEngine> {
    let Some((det, rec)) = models_from_env() else {
        eprintln!(
            "skipping {test}: set OCRS_DETECTION_MODEL and OCRS_RECOGNITION_MODEL \
             (see `make test-capi`)"
        );
        return None;
    };
    let engine = unsafe { ocrs_engine_new(det.as_ptr(), rec.as_ptr()) };
    assert!(
        !engine.is_null(),
        "engine creation failed: {}",
        last_error()
    );
    Some(engine)
}

#[test]
#[ignore = "requires model files; run via `make test-capi`"]
fn get_text_reads_known_image() {
    let Some(engine) = engine_or_skip("get_text_reads_known_image") else {
        return;
    };
    let (pixels, width, height) = load_rgb(TEST_IMAGE);

    let text_ptr = unsafe { ocrs_engine_get_text(engine, pixels.as_ptr(), width, height, 3) };
    assert!(!text_ptr.is_null(), "get_text failed: {}", last_error());
    let text = unsafe { CStr::from_ptr(text_ptr) }
        .to_string_lossy()
        .into_owned();

    unsafe { ocrs_string_free(text_ptr) };
    unsafe { ocrs_engine_free(engine) };

    // Recognition output is not byte-stable across model versions, so assert on
    // a substring the image clearly contains rather than the exact text.
    assert!(
        text.contains("Rust"),
        "expected recognized text to contain \"Rust\", got:\n{text}"
    );
    assert!(text.len() > 50, "expected a substantial amount of text");
}

#[test]
#[ignore = "requires model files; run via `make test-capi`"]
fn get_text_lines_returns_boxes() {
    let Some(engine) = engine_or_skip("get_text_lines_returns_boxes") else {
        return;
    };
    let (pixels, width, height) = load_rgb(TEST_IMAGE);

    let lines_ptr =
        unsafe { ocrs_engine_get_text_lines(engine, pixels.as_ptr(), width, height, 3) };
    assert!(
        !lines_ptr.is_null(),
        "get_text_lines failed: {}",
        last_error()
    );

    let lines = unsafe { &*lines_ptr };
    assert!(
        lines.len > 1,
        "expected multiple text lines, got {}",
        lines.len
    );

    let slice = unsafe { std::slice::from_raw_parts(lines.lines, lines.len) };
    let mut combined = String::new();
    for line in slice {
        assert!(!line.text.is_null(), "line text pointer should be non-NULL");
        let text = unsafe { CStr::from_ptr(line.text) }.to_string_lossy();
        assert!(!text.is_empty(), "line text should not be empty");
        // The bounding box should be non-degenerate.
        assert!(line.right > line.left, "box width should be positive");
        assert!(line.bottom > line.top, "box height should be positive");
        combined.push_str(&text);
        combined.push('\n');
    }

    unsafe { ocrs_text_lines_free(lines_ptr) };
    unsafe { ocrs_engine_free(engine) };

    assert!(
        combined.contains("Rust"),
        "expected combined line text to contain \"Rust\", got:\n{combined}"
    );
}
