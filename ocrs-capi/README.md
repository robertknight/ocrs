# ocrs-capi

A C ABI for the [`ocrs`](../ocrs) OCR engine.

## Available functions

The following functions are generated from the Rust source with
[cbindgen](https://github.com/mozilla/cbindgen).

(Run `make capi-header` after changing the API, or rely on the `header`
test to flag drift.)

- `ocrs_engine_new` / `ocrs_engine_new_from_memory`: create an engine from
  model files or in-memory model buffers.
- `ocrs_engine_get_text`: detect and recognize all text, returned as one UTF-8
  string.
- `ocrs_engine_get_text_lines`: same, but per line with bounding boxes.
- `ocrs_last_error`: thread-local description of the last failure.
- `ocrs_version`, plus `*_free` functions for every owned pointer.

Images are passed as raw 8-bit pixels in row-major, channels-last (HWC) order.
Supported channel counts are 1 (grey), 3 (RGB) and 4 (RGBA). Decoding PNG/JPEG
is the caller's responsibility.

## Building

```sh
# Shared library: target/release/libocrs_capi.{so,dylib} or ocrs_capi.dll
# Static library: target/release/libocrs_capi.a
cargo build -p ocrs-capi --release
```

The library name is `ocrs_capi` (not `ocrs`) (`cdylib` already taken by `ocrs`
crate).

Custom models in ONNX format can be loaded by enabling the `onnx` feature:

```sh
cargo build -p ocrs-capi --release --features onnx
```

## Testing

Run unit tests (arg validation, error handling) without models:

```sh
cargo test -p ocrs-capi
```

Running full detect/recognize pipline, downloading and using model files:

```sh
make test-capi
```

To run them against other models, set `OCRS_DETECTION_MODEL` and
`OCRS_RECOGNITION_MODEL` to absolute paths and run
`cargo test -p ocrs-capi --test ocr -- --ignored`.

## Conventions

- Fallible functions return a pointer and yield `NULL` on failure.
- Use `ocrs_last_error()` for a human-readable message of the last failure
  (thread-local, valid until the next fallible call on the same thread).
- Every pointer returned has a matching `*_free` function and must be released
  through it exactly once. Passing `NULL` to any `*_free` is safe.
- An engine may be used concurrently from multiple threads through a shared
  `const OcrsEngine *` (the OCR calls take `&self` and parallelize internally).
  Do not call `ocrs_engine_free` while another thread is still using the engine.
- Panics are caught at the FFI boundary and turned into a `NULL` return plus a
  last-error message, rather than unwinding into foreign code. This relies on
  the default `panic = "unwind"` strategy. If the library is built with
  `panic = "abort"`, a panic terminates the process instead.

## Example

There is a Zig example in [`examples/zig/`](examples/zig/). It detects and
recognizes text in an image.

1. Download the models:

```sh
curl -O https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.rten
curl -O https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.rten
```

2. Produce raw pixels from any image (here using Python + Pillow). Update the
   `width`/`height`/`channels` constants in `main.zig` to match the output:

```sh
python3 -c "from PIL import Image; im=Image.open('input.png').convert('RGB'); \
open('image.rgb','wb').write(im.tobytes()); print(im.size)"
```

3. Build the static library and link the example against it:

```sh
cargo build -p ocrs-capi --release

# macOS:
zig build-exe examples/zig/main.zig \
-I include \
../target/release/libocrs_capi.a \
-lc -framework Accelerate -framework CoreFoundation -framework Security

# Linux (link libm/pthread/dl as needed by your toolchain):
zig build-exe examples/zig/main.zig -I include \
../target/release/libocrs_capi.a -lc -lm
```

4. Run the test

Run `./main` from the directory containing the `.rten` files and `image.rgb`.

(The same `libocrs_capi.a` plus `include/ocrs.h` can be wired into a `build.zig`
with `exe.addObjectFile`, `exe.addIncludePath` and `exe.linkLibC()`.)

## macOS linking noters

`ocrs` uses [`rten`](https://github.com/robertknight/rten) for inference.
macOS pulls in the `Accelerate`, `CoreFoundation`, and `Security` system
frameworks. When statically linking `libocrs_capi.a`, pass those framework flags
(as seen above). The `cdylib` (`libocrs_capi.dylib`) already has them resolved,
so dynamic linking only needs `-locrs_capi` plus a library search path.
