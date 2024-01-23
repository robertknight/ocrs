use std::collections::VecDeque;
use std::error::Error;
use std::fs;

use ocrs::{OcrEngine, OcrEngineParams};
use rten::Model;
use rten_imageio::read_image;
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
                    bin_name = parser.bin_name().unwrap_or("detr")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let image = values.pop_front().ok_or("missing `image` arg")?;

    Ok(Args { image })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    // Use the `download-models.sh` script to download the models.
    let detection_model_data = fs::read("text-detection.rten")?;
    let rec_model_data = fs::read("text-recognition.rten")?;

    let detection_model = Model::load(&detection_model_data)?;
    let recognition_model = Model::load(&rec_model_data)?;

    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        ..Default::default()
    })?;

    // Read image using image-rs library and convert to a
    // (channels, height, width) tensor with f32 values in [0, 1].
    let image = read_image(&args.image)?;

    // Apply standard image pre-processing expected by this library (convert
    // to greyscale, map range to [-0.5, 0.5]).
    let ocr_input = engine.prepare_input(image.view())?;

    // Phase 1: Detect text words
    let word_rects = engine.detect_words(&ocr_input)?;

    // Phase 2: Perform layout analysis
    let line_rects = engine.find_text_lines(&ocr_input, &word_rects);

    // Phase 3: Recognize text
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
