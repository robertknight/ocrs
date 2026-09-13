//! C ABI for the [`ocrs`] OCR engine.
//!
//! See README.md.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

use ocrs::{ImageSource, OcrEngine, OcrEngineParams, TextItem};
use rten::Model;
use rten_imageproc::BoundingRect;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store `msg` as the calling thread's last error, readable via [`ocrs_last_error`].
fn set_last_error(msg: impl Into<String>) {
    let msg = msg.into();
    let cstr = CString::new(msg).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(cstr));
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Opaque handle to an OCR engine. Create with [`ocrs_engine_new`] or
/// [`ocrs_engine_new_from_memory`] and release with [`ocrs_engine_free`].
pub struct OcrsEngine {
    engine: OcrEngine,
}

/// A recognized line of text together with the axis-aligned bounding box of its
/// characters in the input image (in pixels).
#[repr(C)]
pub struct OcrsTextLine {
    /// NUL-terminated UTF-8 text of the line. Owned by the parent [`OcrsTextLines`].
    pub text: *mut c_char,
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

/// An array of recognized text lines. Release with [`ocrs_text_lines_free`].
#[repr(C)]
pub struct OcrsTextLines {
    pub lines: *mut OcrsTextLine,
    pub len: usize,
}

/// Return the calling thread's most recent error message.
///
/// The returned pointer is owned by the library and remains valid until the
/// next fallible call on the same thread. Returns NULL if there is no error.
#[no_mangle]
pub extern "C" fn ocrs_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match &*slot.borrow() {
        Some(cstr) => cstr.as_ptr(),
        None => ptr::null(),
    })
}

/// Run `f`, converting any panic into a stored error and the given fallback.
fn guard<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(_) => {
            set_last_error("ocrs: caught panic at FFI boundary");
            fallback
        }
    }
}

/// Borrow a NUL-terminated path argument as `&str`.
///
/// `ptr` must be NULL or point to a NUL-terminated C string that outlives the
/// returned reference.
unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

/// Construct an engine from already-loaded models.
fn make_engine(
    detection_model: Option<Model>,
    recognition_model: Option<Model>,
) -> Result<*mut OcrsEngine, String> {
    let engine = OcrEngine::new(OcrEngineParams {
        detection_model,
        recognition_model,
        ..Default::default()
    })
    .map_err(|err| format!("failed to create engine: {err}"))?;
    Ok(Box::into_raw(Box::new(OcrsEngine { engine })))
}

/// Turn an engine-constructor result into a raw pointer, recording any error
/// message for [`ocrs_last_error`] and returning NULL on failure.
fn finish_engine(result: Result<*mut OcrsEngine, String>) -> *mut OcrsEngine {
    match result {
        Ok(engine) => {
            clear_last_error();
            engine
        }
        Err(msg) => {
            set_last_error(msg);
            ptr::null_mut()
        }
    }
}

/// Load a model from a file path, returning `Ok(None)` when `ptr` is NULL.
///
/// `ptr` must be NULL or point to a NUL-terminated C string.
unsafe fn load_model_file(ptr: *const c_char, kind: &str) -> Result<Option<Model>, String> {
    if ptr.is_null() {
        return Ok(None);
    }
    let path = cstr_to_str(ptr).ok_or_else(|| format!("{kind} model path is not valid UTF-8"))?;
    Model::load_file(path)
        .map(Some)
        .map_err(|err| format!("failed to load {kind} model '{path}': {err}"))
}

/// Load a model from a memory buffer, returning `Ok(None)` when `ptr` is NULL.
///
/// `ptr` must be NULL or point to at least `len` readable bytes.
unsafe fn load_model_memory(
    ptr: *const u8,
    len: usize,
    kind: &str,
) -> Result<Option<Model>, String> {
    if ptr.is_null() {
        return Ok(None);
    }
    let bytes = slice::from_raw_parts(ptr, len).to_vec();
    Model::load(bytes)
        .map(Some)
        .map_err(|err| format!("failed to load {kind} model from memory: {err}"))
}

/// Create an engine, loading the detection and recognition models from files.
///
/// `detection_model_path` and `recognition_model_path` are NUL-terminated paths
/// to `.onnx` model files. Either may be NULL to omit that model, though
/// detection is required for [`ocrs_engine_get_text`] and recognition is
/// required for any text output.
///
/// Returns NULL on failure. See [`ocrs_last_error`].
///
/// # Safety
///
/// Each non-NULL path argument must point to a NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn ocrs_engine_new(
    detection_model_path: *const c_char,
    recognition_model_path: *const c_char,
) -> *mut OcrsEngine {
    guard(ptr::null_mut(), || {
        finish_engine((|| {
            let detection = load_model_file(detection_model_path, "detection")?;
            let recognition = load_model_file(recognition_model_path, "recognition")?;
            make_engine(detection, recognition)
        })())
    })
}

/// Create an engine from models held in memory.
///
/// Each model is read from a buffer of `*_len` bytes. A NULL pointer (with any
/// length) omits that model. The buffers are only read during this call and may
/// be freed by the caller afterwards.
///
/// Returns NULL on failure. See [`ocrs_last_error`].
///
/// # Safety
///
/// Each non-NULL model pointer must point to at least `*_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn ocrs_engine_new_from_memory(
    detection_model: *const u8,
    detection_model_len: usize,
    recognition_model: *const u8,
    recognition_model_len: usize,
) -> *mut OcrsEngine {
    guard(ptr::null_mut(), || {
        finish_engine((|| {
            let detection = load_model_memory(detection_model, detection_model_len, "detection")?;
            let recognition =
                load_model_memory(recognition_model, recognition_model_len, "recognition")?;
            make_engine(detection, recognition)
        })())
    })
}

/// Free an engine created by `ocrs_engine_new*`. Passing NULL is a no-op.
///
/// # Safety
///
/// `engine` must be NULL or a pointer returned by `ocrs_engine_new*` that has
/// not already been freed.
#[no_mangle]
pub unsafe extern "C" fn ocrs_engine_free(engine: *mut OcrsEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

/// Borrow the image bytes and prepare OCR input, setting the last error on
/// failure.
unsafe fn prepare_input(
    engine: &OcrsEngine,
    image_data: *const u8,
    width: u32,
    height: u32,
    channels: u32,
) -> Option<ocrs::OcrInput> {
    if image_data.is_null() {
        set_last_error("image_data is NULL");
        return None;
    }
    if !matches!(channels, 1 | 3 | 4) {
        set_last_error(format!(
            "unsupported channel count {channels}; expected 1, 3 or 4"
        ));
        return None;
    }
    if width == 0 || height == 0 {
        set_last_error("image width and height must be non-zero");
        return None;
    }
    let len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(channels as usize));
    let Some(len) = len else {
        set_last_error("image dimensions overflow");
        return None;
    };

    let bytes = slice::from_raw_parts(image_data, len);
    let source = match ImageSource::from_bytes(bytes, (width, height)) {
        Ok(source) => source,
        Err(err) => {
            set_last_error(format!("invalid image: {err}"));
            return None;
        }
    };
    match engine.engine.prepare_input(source) {
        Ok(input) => Some(input),
        Err(err) => {
            set_last_error(format!("failed to prepare image: {err}"));
            None
        }
    }
}

/// Detect and recognize all text in an image and return it as a single
/// NUL-terminated UTF-8 string, with lines separated by `\n`.
///
/// `image_data` points to `width * height * channels` bytes in HWC order.
/// The engine must have both a detection and a recognition model.
///
/// Returns NULL on failure. See [`ocrs_last_error`]. Free the result with
/// [`ocrs_string_free`].
///
/// # Safety
///
/// `engine` must be a valid engine pointer and `image_data` must point to at
/// least `width * height * channels` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn ocrs_engine_get_text(
    engine: *const OcrsEngine,
    image_data: *const u8,
    width: u32,
    height: u32,
    channels: u32,
) -> *mut c_char {
    guard(ptr::null_mut(), || {
        let Some(engine) = engine.as_ref() else {
            set_last_error("engine is NULL");
            return ptr::null_mut();
        };
        let Some(input) = prepare_input(engine, image_data, width, height, channels) else {
            return ptr::null_mut();
        };
        match engine.engine.get_text(&input) {
            Ok(text) => match CString::new(text) {
                Ok(cstr) => {
                    clear_last_error();
                    cstr.into_raw()
                }
                Err(_) => {
                    set_last_error("recognized text contained an interior NUL byte");
                    ptr::null_mut()
                }
            },
            Err(err) => {
                set_last_error(format!("OCR failed: {err}"));
                ptr::null_mut()
            }
        }
    })
}

/// Detect and recognize text, returning per-line text and bounding boxes.
///
/// `image_data` points to `width * height * channels` bytes in HWC order. The
/// engine must have both a detection and a recognition model. Lines with no
/// recognized characters are omitted.
///
/// Returns NULL on failure. See [`ocrs_last_error`]. Free the result with
/// [`ocrs_text_lines_free`].
///
/// # Safety
///
/// `engine` must be a valid engine pointer and `image_data` must point to at
/// least `width * height * channels` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn ocrs_engine_get_text_lines(
    engine: *const OcrsEngine,
    image_data: *const u8,
    width: u32,
    height: u32,
    channels: u32,
) -> *mut OcrsTextLines {
    guard(ptr::null_mut(), || {
        let Some(engine) = engine.as_ref() else {
            set_last_error("engine is NULL");
            return ptr::null_mut();
        };
        let Some(input) = prepare_input(engine, image_data, width, height, channels) else {
            return ptr::null_mut();
        };

        let words = match engine.engine.detect_words(&input) {
            Ok(words) => words,
            Err(err) => {
                set_last_error(format!("word detection failed: {err}"));
                return ptr::null_mut();
            }
        };
        let line_rects = engine.engine.find_text_lines(&input, &words);
        let recognized = match engine.engine.recognize_text(&input, &line_rects) {
            Ok(lines) => lines,
            Err(err) => {
                set_last_error(format!("text recognition failed: {err}"));
                return ptr::null_mut();
            }
        };

        let mut lines: Vec<OcrsTextLine> = Vec::new();
        for line in recognized.into_iter().flatten() {
            let text = line.to_string();
            let cstr = match CString::new(text) {
                Ok(cstr) => cstr,
                Err(_) => {
                    // Skip lines we cannot represent as a C string rather than
                    // failing the whole call.
                    continue;
                }
            };
            let rect = line.rotated_rect().bounding_rect();
            lines.push(OcrsTextLine {
                text: cstr.into_raw(),
                left: rect.left(),
                top: rect.top(),
                right: rect.right(),
                bottom: rect.bottom(),
            });
        }

        clear_last_error();
        let len = lines.len();
        let ptr = if len == 0 {
            ptr::null_mut()
        } else {
            let boxed = lines.into_boxed_slice();
            Box::into_raw(boxed) as *mut OcrsTextLine
        };
        Box::into_raw(Box::new(OcrsTextLines { lines: ptr, len }))
    })
}

/// Free a string returned by [`ocrs_engine_get_text`]. Passing NULL is a no-op.
///
/// # Safety
///
/// `text` must be NULL or a pointer returned by [`ocrs_engine_get_text`] that
/// has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn ocrs_string_free(text: *mut c_char) {
    if !text.is_null() {
        drop(CString::from_raw(text));
    }
}

/// Free a result returned by [`ocrs_engine_get_text_lines`], including every
/// line's text. Passing NULL is a no-op.
///
/// # Safety
///
/// `lines` must be NULL or a pointer returned by [`ocrs_engine_get_text_lines`]
/// that has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn ocrs_text_lines_free(lines: *mut OcrsTextLines) {
    if lines.is_null() {
        return;
    }
    let owned = Box::from_raw(lines);
    if !owned.lines.is_null() && owned.len > 0 {
        let slice = slice::from_raw_parts_mut(owned.lines, owned.len);
        let boxed: Box<[OcrsTextLine]> = Box::from_raw(slice);
        for line in boxed.iter() {
            if !line.text.is_null() {
                drop(CString::from_raw(line.text));
            }
        }
    }
}

/// Return the ocrs version as a static NUL-terminated string.
#[no_mangle]
pub extern "C" fn ocrs_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read the calling thread's last error as an owned `String`.
    ///
    /// Each `#[test]` runs on its own thread, so the thread-local error state is
    /// isolated between tests.
    fn last_error() -> Option<String> {
        let ptr = ocrs_last_error();
        if ptr.is_null() {
            None
        } else {
            Some(
                unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }

    /// Create an engine with no models loaded. Engine construction itself
    /// succeeds. Only later detection/recognition calls require models.
    fn engine_without_models() -> *mut OcrsEngine {
        let engine = unsafe { ocrs_engine_new(ptr::null(), ptr::null()) };
        assert!(!engine.is_null(), "{:?}", last_error());
        engine
    }

    #[test]
    fn version_matches_package() {
        let ptr = ocrs_version();
        assert!(!ptr.is_null());
        let version = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn new_with_null_paths_succeeds() {
        let engine = engine_without_models();
        assert!(last_error().is_none());
        unsafe { ocrs_engine_free(engine) };
    }

    #[test]
    fn new_with_bad_path_reports_error() {
        let path = CString::new("/no/such/model.onnx").unwrap();
        let engine = unsafe { ocrs_engine_new(path.as_ptr(), ptr::null()) };
        assert!(engine.is_null());
        let err = last_error().expect("error should be set");
        assert!(err.contains("detection model"), "{err}");
    }

    #[test]
    fn new_from_memory_with_garbage_reports_error() {
        let garbage = [0u8; 16];
        let engine =
            unsafe { ocrs_engine_new_from_memory(garbage.as_ptr(), garbage.len(), ptr::null(), 0) };
        assert!(engine.is_null());
        assert!(last_error()
            .unwrap()
            .contains("detection model from memory"));
    }

    #[test]
    fn get_text_without_models_reports_error() {
        let engine = engine_without_models();
        let pixels = [0u8; 4 * 4 * 3];
        let text = unsafe { ocrs_engine_get_text(engine, pixels.as_ptr(), 4, 4, 3) };
        assert!(text.is_null());
        assert!(last_error().is_some());
        unsafe { ocrs_engine_free(engine) };
    }

    #[test]
    fn get_text_rejects_null_engine() {
        let pixels = [0u8; 3];
        let text = unsafe { ocrs_engine_get_text(ptr::null(), pixels.as_ptr(), 1, 1, 3) };
        assert!(text.is_null());
        assert_eq!(last_error().as_deref(), Some("engine is NULL"));
    }

    #[test]
    fn get_text_rejects_null_image() {
        let engine = engine_without_models();
        let text = unsafe { ocrs_engine_get_text(engine, ptr::null(), 1, 1, 3) };
        assert!(text.is_null());
        assert_eq!(last_error().as_deref(), Some("image_data is NULL"));
        unsafe { ocrs_engine_free(engine) };
    }

    #[test]
    fn get_text_rejects_bad_channel_count() {
        let engine = engine_without_models();
        let pixels = [0u8; 2 * 2 * 2];
        let text = unsafe { ocrs_engine_get_text(engine, pixels.as_ptr(), 2, 2, 2) };
        assert!(text.is_null());
        assert!(last_error().unwrap().contains("channel count"));
        unsafe { ocrs_engine_free(engine) };
    }

    #[test]
    fn get_text_rejects_zero_dimensions() {
        let engine = engine_without_models();
        let pixels = [0u8; 3];
        let text = unsafe { ocrs_engine_get_text(engine, pixels.as_ptr(), 0, 1, 3) };
        assert!(text.is_null());
        assert!(last_error().unwrap().contains("non-zero"));
        unsafe { ocrs_engine_free(engine) };
    }

    #[test]
    fn free_functions_accept_null() {
        unsafe {
            ocrs_engine_free(ptr::null_mut());
            ocrs_string_free(ptr::null_mut());
            ocrs_text_lines_free(ptr::null_mut());
        }
    }
}
