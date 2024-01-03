.PHONY: build
build:
	cargo build

.PHONY: clean
clean:
	rm -rf dist/*
	rm -rf target/

.PHONY: check
check: checkformatting test lint

.PHONY: checkformatting
checkformatting:
	cargo fmt --check

.PHONY: lint
lint:
	cargo clippy --workspace

.PHONY: test
test:
	cargo test --workspace

.PHONY: test-e2e
test-e2e:
	# TODO - Actually check the output is correct
	cargo run --release -p ocrs-cli ocrs-cli/test-data/why-rust.png

.PHONY: wasm
wasm:
	RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown --package ocrs
	wasm-bindgen target/wasm32-unknown-unknown/release/ocrs.wasm --out-dir dist/ --target web --reference-types --weak-refs
	tools/optimize-wasm.sh dist/ocrs_bg.wasm

.PHONY: wasm-all
wasm-all: wasm wasm-nosimd
