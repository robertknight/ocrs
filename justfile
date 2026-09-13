# Build all crates in the workspace.
build:
    cargo build

# Remove build outputs.
clean:
    rm -rf js/dist/*
    rm -rf target/

# Run formatting, test and lint checks.
check: checkformatting test lint

# Check that source files are formatted.
checkformatting:
    cargo fmt --check

# Build API documentation.
doc:
    cargo doc

# Download models and run the `hello_ocr` example.
example:
    cd ocrs/examples && ./download-models.sh
    cargo run -p ocrs --release --example hello_ocr ocrs/examples/rust-book.jpg

# Run lint checks.
lint:
    cargo clippy --workspace

# The `header` test checks that the committed header is up to date.

# Regenerate the ocrs-capi C header from the Rust source.
capi-header:
    OCRS_UPDATE_HEADER=1 cargo test -p ocrs-capi --test header

# Run unit tests.
test:
    cargo test --workspace

# These need the model files and so are skipped by `just test`. Models are
# downloaded first, and the tests run in release mode.

# Run ocrs-capi integration tests (full OCR pipeline through the C ABI).
test-capi:
    cd ocrs/examples && ./download-models.sh
    OCRS_DETECTION_MODEL={{justfile_directory()}}/ocrs/examples/text-detection.onnx \
    OCRS_RECOGNITION_MODEL={{justfile_directory()}}/ocrs/examples/text-recognition.onnx \
    cargo test -p ocrs-capi --release --test ocr -- --ignored

# Run end-to-end tests against the CLI.
test-e2e:
    python tools/test-e2e.py ocrs-cli/test-data/

# Update the expected output of the end-to-end tests.
update-e2e:
    python tools/test-e2e.py --update ocrs-cli/test-data/

# Build the ocrs library for the browser.
wasm:
    RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown --package ocrs
    wasm-bindgen target/wasm32-unknown-unknown/release/ocrs.wasm --out-dir js/dist/ --target web --reference-types --weak-refs
    tools/optimize-wasm.sh js/dist/ocrs_bg.wasm

# Run the result using:
#
#   wasmtime --dir . target/wasm32-wasi/release/ocrs.wasm --detect-model text-detection.onnx --rec-model text-recognition.onnx ocrs-cli/test-data/why-rust.png

# Build Ocrs CLI for non-browser WebAssembly runtimes (eg. wasmtime).
wasm-wasi:
    RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-wasi --package ocrs-cli
