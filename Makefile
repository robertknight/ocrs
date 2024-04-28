.PHONY: build
build:
	cargo build

.PHONY: clean
clean:
	rm -rf js/dist/*
	rm -rf target/

.PHONY: check
check: checkformatting test lint

.PHONY: checkformatting
checkformatting:
	cargo fmt --check

.PHONY: docs
doc:
	cargo doc

.PHONY: run-example
example:
	cd ocrs/examples && ./download-models.sh
	cargo run -p ocrs --release --example hello_ocr ocrs/examples/rust-book.jpg

.PHONY: lint
lint:
	cargo clippy --workspace

.PHONY: test
test:
	cargo test --workspace

.PHONY: test-e2e
test-e2e:
	python tools/test-e2e.py ocrs-cli/test-data/

.PHONY: wasm
wasm:
	RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown --package ocrs
	wasm-bindgen target/wasm32-unknown-unknown/release/ocrs.wasm --out-dir js/dist/ --target web --reference-types --weak-refs
	tools/optimize-wasm.sh js/dist/ocrs_bg.wasm

.PHONY: wasm-all
wasm-all: wasm wasm-nosimd
