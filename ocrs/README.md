# ocrs

This crate contains the **ocrs** OCR engine as a library. See the
[main project README][main_readme] for general details about the project.

[main_readme]: https://github.com/robertknight/ocrs/blob/main/README.md

## Usage

See [examples/hello_ocr.rs]() for a minimal example of using this library in
a Rust application.

```sh
cd examples/

# Download models in .rten format.
./download-models.sh

# Run OCR on an image and print the extracted text.
cargo run -r --example hello_ocr rust-book.jpg
```
