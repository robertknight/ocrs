# ocrs browser extension

The Ocrs browser extension allows you to copy text from images, videos, PDFs or
any other content displayed in a browser tab.

It has currently only been tested in Chrome.

## Building the extension

1.  First, build the WebAssembly OCR library. In the root directory of this
    repository run:

    ```sh
    make wasm
    ```

2.  Download pre-trained models. The easiest way to do this is to run the
    ocrs CLI tool, which will download models from the preferred location, and
    then copy them from the cache directory to `<repo_root>/models/ocr`.
    In the root of the repository run:

    ```sh
    cargo run -r -p ocrs-cli test-image.jpeg
    mkdir -p models/ocr
    cp ~/.cache/ocrs/text-detection.rten ~/.cache/ocrs/text-recognition.rten models/ocr
    ```

    Where `test-image.jpeg` can be any image you have available.

3.  Navigate to this directory and build the browser extension:

    ```sh
    cd ocrs-extension
    npm install
    make build
    ```

4.  In Chrome, go to `chrome://extensions` and select "Load unpacked extension",
    then select the `ocrs-extension` directory.

## Using the extension

1.  After installing the extension, click the puzzle piece icon in Chrome's
    toolbar and click the pin icon next to Ocrs to add it to the toolbar.

2.  On any browser tab, click the Ocrs logo in the toolbar to take a screenshot
    of the current tab and highlight selectable text.

3.  Click anywhere outside of a text region or press Escape to close the OCR
    overlay.
