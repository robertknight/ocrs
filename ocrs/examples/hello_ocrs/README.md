# Hello ocrs

## Performance note

Make sure you build the `ocrs` crate and its `rten*` dependencies in release mode. Debug builds of these crates will be extremely slow.

## Download models

```shell 
# Download models in .rten format.
./download-models.sh
```

## Usage
```shell 
# Run OCR on an image and print the extracted text.
cargo run --release rust-book.jpg
```

