# ocrs-cli

CLI tool for extracting text from images using the
[**ocrs**](https://github.com/robertknight/rten/tree/main/ocrs) OCR engine.

## Installation

These steps assume you have [Rust and cargo](https://www.rust-lang.org/tools/install) installed.

```sh
cargo install ocrs
```

## Usage

Extract text from an image and print it to standard output:

```sh
ocrs image.jpeg
```

The first time that you run the tool it will download default models for text
detection and recognition. You can override this by using the `--detect-model
<path>` and `--rec-model <path>` flags to set the path to the models.

### Layout export

By default ocrs outputs just the extracted text. Specifying the `--export-boxes
path.json` flag will cause text layout information to be produced in JSON
format.

### Debug output

If the `--debug` flag is enabled, ocrs will create an annotated image in the
current directory showing where words and lines were detected, as well as
logging performance information.

## Language support

The current version of this tool was trained primarily on English text.
Support for more languages is planned for the future.
