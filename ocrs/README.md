# ocrs

This crate contains the **ocrs** OCR engine as a library. See the
[main project README][main_readme] for general details about the project.

[main_readme]: https://github.com/robertknight/ocrs/blob/main/README.md

## Performance note

Make sure you build the `ocrs` crate and its `rten*` dependencies in **release
mode**. Debug builds of these crates will be extremely slow.

## Usage

See [examples/hello_ocr.rs](./examples/hello_ocr.rs) for a minimal example of using this library in
a Rust application.

```sh
cd examples/

# Download models in .rten format.
./download-models.sh

# Run OCR on an image and print the extracted text.
cargo run --release --example hello_ocr rust-book.jpg
```

> Note that, performance of the binary will differ significantly if `debug`
> build profile is used. Learn more: https://github.com/robertknight/ocrs/issues/117#issuecomment-2362314977
