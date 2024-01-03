# Ocrs

**ocrs** is a new OCR engine, written in Rust. It can be used as a CLI tool and
as a library.

The goal is to create a modern OCR engine that:

 - Works well on a wide variety of images (scanned documents, photos containing
   text, screenshots etc.) with zero or much less preprocessing effort compared
   to earlier engines like [Tesseract][tesseract]. This is achieved by using
   machine learning more extensively in the pipeline.
 - Is easy to compile and run across a variety of platforms, including
   WebAssembly
 - Is trained on open and liberally licensed datasets
 - Has a codebase that is easy to understand and modify

Under the hood, the library uses neural network models trained in
[PyTorch][pytorch], which are then exported to [ONNX][onnx] and executed using
the [RTen][rten] engine. See the [models](#models-and-datasets) section for
more details.

[onnx]: https://onnx.ai
[pytorch]: https://pytorch.org
[rten]: https://github.com/robertknight/rten
[tesseract]: https://github.com/tesseract-ocr/tesseract

## Status

ocrs is currently in an early preview. Expect more errors than commercial OCR
engines.

## CLI installation

To install the CLI tool, you will first need Rust and Cargo installed. Then
run:

```sh
$ cargo install ocrs-cli
```

## CLI usage:

To extract text from an image, run:

```sh
$ ocrs image.png
```

When the tool is run for the first time, it will download the required models
automatically and store them in `~/.cache/ocrs`.

### Additional examples

Extract text from an image and write to `content.txt`:

```sh
$ ocrs image.png -o content.txt
```

Extract text and layout information from the image in JSON format:

```sh
$ ocrs image.png --json -o content.json
```

Annotate an image to show the location of detected words and lines:

```sh
$ ocrs image.png --png -o annotated.png
````

## Models and datasets

ocrs uses neural network models written in PyTorch. See the
[ocrs-models](https://github.com/robertknight/ocrs-models) repository for more
details about the models and datasets, as well as tools for training custom
models. These models are also available in ONNX format for use with other
machine learning runtimes.

## Development

To build and run the ocrs library and CLI tool locally you will need a recent
stable Rust version installed. Then run:

```sh
git clone https://github.com/robertknight/ocrs.git
cd ocrs
cargo run -p ocrs-cli -r -- image.png
```

### Library installation

See the [ocrs crate README](ocrs/) for details on how to use ocrs as a Rust
library.
