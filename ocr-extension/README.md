# Pixel Reader

Pixel Reader is a browser extension that enables you to copy text from images,
videos, PDFs or any other content displayed in the current tab.

## Building the extension

 1. First, build the WebAssembly OCR library and download pre-trained models.
    (TODO: Add details on where to get models from). In the root directory of
    the repository run:

    ```sh
    make wasm-ocr
    ```

 2. Navigate to this directory and build the browser extension:

    ```sh
    cd ocr-extension
    npm install
    make build
    ```

 3. In Chrome, go to `chrome://extensions` and select "Load unpacked extension",
    then select the `ocr-extension` directory.
 
## Using the extension

 1. After installing the extension, click the puzzle piece icon in Chrome's
    toolbar and click the pin icon next to Pixel Reader to add it to the
    toolbar.

 2. On any browser tab, click the Pixel Reader logo in the toolbar to take
    a screenshot of the current tab and highlight selectable text.

 3. Click anywhere outside of a text region or press Escape to close the OCR
    overlay.
