# ocrs

**ocrs** is an OCR engine written in Rust. It extracts text, including layout
information, from images. It uses the [RTen](https://github.com/robertknight/rten)
machine learning runtime.

ocrs uses neural network models written in PyTorch. See the
[ocrs-models](https://github.com/robertknight/ocrs-models) repository for more
details and tools for training custom models. These models are also available in
ONNX format for use with other machine learning runtimes.

## Background and status

The original goal of this project was to create a more modern alternative to
Tesseract, which is still (!) the de-facto open-source OCR engine. See
https://github.com/robertknight/tesseract-wasm/issues/87 for context. Some ways
in which this project aims to improve upon Tesseract are:

- Reducing the need to manually clean up and pre-process images before feeding
  them into engine. This is achieved by replacing manually written components
  for text detection and, in future, layout analysis, with machine learning
  models.
- Using modern tools (eg. PyTorch) for training. Together with use of open
  datasets this should make it easier to recreate and customize models
- Making the models available in portable formats (ONNX) which can be used with
  many different runtimes
- Better support for Unicode and large character sets.

ocrs is currently in an early preview. Expect more errors than commercial OCR
engines. The layout analysis phase is still uses manually written code which
can be brittle.

## Command-line usage

This library is wrapped in a CLI tool in the [ocrs-cli](../ocrs-cli) crate,
which you can use as follows:

```sh
$ cargo install ocrs-cli
$ ocrs image.png
```

See the [ocrs-cli README](../ocrs-cli/README.md) for more details.

## Library usage

See [examples/hello_ocr.rs]() for a minimal example of using this library in
a Rust application.

```sh
cd examples/

# Download models in .rten format.
./download-models.sh

# Run OCR on an image and print the extracted text.
cargo run -r --example hello_ocr rust-book.jpg
```
