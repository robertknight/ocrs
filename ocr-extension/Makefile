.PHONY: build
build:
	mkdir -p build
	cp ../dist/wasnn_ocr.js ../dist/wasnn_ocr.d.ts build/
	cp ../dist/wasnn_ocr_bg.wasm build/
	cp ../models/ocr/text-detection.model build/
	cp ../models/ocr/text-recognition.model build/
	npm run build

.PHONY: clean
clean:
	rm -rf build build-extension
