# ocrs

**ocrs** is an OCR library written in Rust. It extracts text, including layout
information, from images. It uses the [RTen](https://github.com/robertknight/rten)
machine learning runtime.

This library is wrapped in a CLI tool in the [ocrs-cli](../ocrs-cli) crate,
which you can use as follows:

```sh
$ cargo install ocrs-cli
$ ocrs image.png
```

See the [main.rs](../ocrs-cli/src/main.rs) module in that crate for steps to
use the library.

## Usage

See [examples/hello_ocr.rs]() for a minimal example.

```sh
cd examples/

# Download models in .rten format.
./download-models.sh

# Run OCR on an image and print the extracted text.
cargo run -r --example hello_ocr rust-book.jpg
```
