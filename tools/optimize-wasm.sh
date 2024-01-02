#!/bin/sh

set -eu

BIN_PATH="${1:-}"
WASMOPT_BIN=$(which wasm-opt || true)

if [ -z "$BIN_PATH" ]; then
  echo "Usage: $(basename "$0") <WASM binary>"
  exit 1
fi

if [ -z "$WASMOPT_BIN" ]; then
  echo 'Skipping post-compilation optimization because `wasm-opt` binary was not found.'
  exit
fi

if [ -n "${SKIP_WASM_OPT:-}" ]; then
  echo  "Skipping post-compilation optimization because SKIP_WASM_OPT is set"
  exit
fi

wasm-opt --enable-simd --enable-reference-types -O2 "$BIN_PATH" -o "$BIN_PATH".optimized
mv "$BIN_PATH.optimized" "$BIN_PATH"
