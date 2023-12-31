.PHONY: build
build:
	mkdir -p build
	cp ../dist/ocrs.js ../dist/ocrs.d.ts build/
	cp ../dist/ocrs_bg.wasm build/
	cp ../models/ocr/text-detection.model build/
	cp ../models/ocr/text-recognition.model build/
	npm run build

.PHONY: clean
clean:
	rm -rf build build-extension
