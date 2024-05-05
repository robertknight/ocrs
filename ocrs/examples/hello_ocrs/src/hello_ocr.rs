use std::collections::VecDeque;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten::Model;
#[allow(unused)]
use rten_tensor::prelude::*;

struct Args {
    image: String,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut parser = lexopt::Parser::from_env();

    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("help") => {
                println!(
                    "Usage: {bin_name} <image>",
                    bin_name = parser.bin_name().unwrap_or("hello_ocrs")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let image = values.pop_front().ok_or("missing `image` arg")?;

    Ok(Args { image })
}

/// Read a file from a path that is relative to the crate root.
fn read_file(path: &str) -> Result<Vec<u8>, std::io::Error> {
    let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    abs_path.push(path);
    fs::read(abs_path)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    // Use the `download-models.sh` script to download the models.
    let detection_model_data = read_file("text-detection.rten")?;
    let rec_model_data = read_file("text-recognition.rten")?;

    let detection_model = Model::load(&detection_model_data)?;
    let recognition_model = Model::load(&rec_model_data)?;

    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        ..Default::default()
    })?;

    // Read image using image-rs library, and convert to RGB if not already
    // in that format.
    let img = image::open(args.image).map(|image| image.into_rgb8())?;

    // Apply standard image pre-processing expected by this library (convert
    // to greyscale, map range to [-0.5, 0.5]).
    let img_source = ImageSource::from_bytes(&img, img.dimensions())?;
    let ocr_input = engine.prepare_input(img_source)?;

    // Detect and recognize text. If you only need the text and don't need any
    // layout information, you can also use `engine.get_text(&ocr_input)`,
    // which returns all the text in an image as a single string.

    // Get oriented bounding boxes of text words in input image.
    let word_rects = engine.detect_words(&ocr_input)?;

    // Group words into lines. Each line is represented by a list of word
    // bounding boxes.
    let line_rects = engine.find_text_lines(&ocr_input, &word_rects);

    // Recognize the characters in each line.
    let line_texts = engine.recognize_text(&ocr_input, &line_rects)?;

    for line in line_texts
        .iter()
        .flatten()
        // Filter likely spurious detections. With future model improvements
        // this should become unnecessary.
        .filter(|l| l.to_string().len() > 1)
    {
        println!("{}", line);
    }

    Ok(())
}
