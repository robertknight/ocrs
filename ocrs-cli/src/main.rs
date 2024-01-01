use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::BufWriter;

use ocrs::{DecodeMethod, OcrEngine, OcrEngineParams};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

mod models;
use models::{load_model, ModelSource};
mod output;
use output::{
    format_json_output, format_text_output, generate_annotated_png, FormatJsonArgs,
    GeneratePngArgs, OutputFormat,
};

/// Read an image from `path` into a CHW tensor.
fn read_image(path: &str) -> Result<NdTensor<f32, 3>, Box<dyn Error>> {
    let input_img = image::open(path)?;
    let input_img = input_img.into_rgb8();

    let (width, height) = input_img.dimensions();

    let in_chans = 3;
    let mut float_img = NdTensor::zeros([in_chans, height as usize, width as usize]);
    for c in 0..in_chans {
        let mut chan_img = float_img.slice_mut([c]);
        for y in 0..height {
            for x in 0..width {
                chan_img[[y as usize, x as usize]] = input_img.get_pixel(x, y)[c] as f32 / 255.0
            }
        }
    }
    Ok(float_img)
}

/// Write a CHW image to a PNG file in `path`.
fn write_image(path: &str, img: NdTensorView<f32, 3>) -> Result<(), Box<dyn Error>> {
    let img_width = img.size(2);
    let img_height = img.size(1);
    let color_type = match img.size(0) {
        1 => png::ColorType::Grayscale,
        3 => png::ColorType::Rgb,
        4 => png::ColorType::Rgba,
        _ => return Err("Unsupported channel count".into()),
    };

    let hwc_img = img.permuted([1, 2, 0]); // CHW => HWC

    let out_img = image_from_tensor(hwc_img);
    let file = fs::File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, img_width as u32, img_height as u32);
    encoder.set_color(color_type);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&out_img)?;

    Ok(())
}

/// Convert an CHW float tensor with values in the range [0, 1] to `Vec<u8>`
/// with values scaled to [0, 255].
fn image_from_tensor(tensor: NdTensorView<f32, 3>) -> Vec<u8> {
    tensor
        .iter()
        .map(|x| (x.clamp(0., 1.) * 255.0) as u8)
        .collect()
}

struct Args {
    /// Path to a text detection model.
    detection_model: Option<String>,

    /// Path to a text recognition model.
    recognition_model: Option<String>,

    /// Path to image to process.
    image: String,

    /// Enable debug output.
    debug: bool,

    output_format: OutputFormat,

    /// Output file path. Defaults to stdout.
    output_path: Option<String>,

    /// Use beam search for sequence decoding.
    beam_search: bool,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut beam_search = false;
    let mut debug = false;
    let mut detection_model = None;
    let mut output_format = OutputFormat::Text;
    let mut output_path = None;
    let mut recognition_model = None;

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("beam") => {
                beam_search = true;
            }
            Long("debug") => {
                debug = true;
            }
            Long("detect-model") => {
                detection_model = Some(parser.value()?.string()?);
            }
            Short('j') | Long("json") => {
                output_format = OutputFormat::Json;
            }
            Short('o') | Long("output") => {
                output_path = Some(parser.value()?.string()?);
            }
            Short('p') | Long("png") => {
                output_format = OutputFormat::Png;
            }
            Long("rec-model") => {
                recognition_model = Some(parser.value()?.string()?);
            }
            Long("help") => {
                println!(
                    "Extract text from an image.

Usage: {bin_name} [OPTIONS] <image>

Options:

  --debug

    Enable debug output.

  --detect-model <path>

    Use a custom text detection model.

  -j, --json

    Output text and structure in JSON format.

  -o, --output <path>

    Output file path (defaults to stdout)

  -p, --png

    Output annotated copy of input image in PNG format.

  --rec-model <path>

    Use a custom text recognition model.

Advanced options:

  --beam

    Use beam search for decoding.
",
                    bin_name = parser.bin_name().unwrap_or("ocrs")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    Ok(Args {
        beam_search,
        debug,
        detection_model,
        output_format,
        output_path,
        image: values.pop_front().ok_or("missing `<image>` arg")?,
        recognition_model,
    })
}

/// Adds context to an error reading or parsing a file.
trait FileErrorContext<T> {
    /// If `self` represents a failed operation to read a file, convert the
    /// error to a message of the form "{context} from {path}: {original_error}".
    fn file_error_context<P: fmt::Display>(self, context: &str, path: P) -> Result<T, String>;
}

impl<T, E: std::fmt::Display> FileErrorContext<T> for Result<T, E> {
    fn file_error_context<P: fmt::Display>(self, context: &str, path: P) -> Result<T, String> {
        self.map_err(|err| format!("{} from \"{}\": {}", context, path, err))
    }
}

/// Default text detection model.
const DETECTION_MODEL: &str =
    "https://s3.amazonaws.com/io.github.robertknight/ocrs-models/text-detection.rten";

/// Default text recognition model.
const RECOGNITION_MODEL: &str =
    "https://s3.amazonaws.com/io.github.robertknight/ocrs-models/text-recognition.rten";

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    // Fetch and load ML models.
    let detection_model_src = args
        .detection_model
        .as_ref()
        .map(|path| ModelSource::Path(path))
        .unwrap_or(ModelSource::Url(DETECTION_MODEL));
    let detection_model = load_model(detection_model_src)
        .file_error_context("Failed to load text detection model", detection_model_src)?;

    let recognition_model_src = args
        .recognition_model
        .as_ref()
        .map(|path| ModelSource::Path(path))
        .unwrap_or(ModelSource::Url(RECOGNITION_MODEL));
    let recognition_model = load_model(recognition_model_src).file_error_context(
        "Failed to load text recognition model",
        recognition_model_src,
    )?;

    // Read image into CHW tensor.
    let color_img =
        read_image(&args.image).file_error_context("Failed to read image", &args.image)?;

    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        debug: args.debug,
        decode_method: if args.beam_search {
            DecodeMethod::BeamSearch { width: 100 }
        } else {
            DecodeMethod::Greedy
        },
    })?;

    let ocr_input = engine.prepare_input(color_img.view())?;
    let word_rects = engine.detect_words(&ocr_input)?;
    let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
    let line_texts = engine.recognize_text(&ocr_input, &line_rects)?;

    let write_output_str = |content: String| -> Result<(), Box<dyn Error>> {
        if let Some(output_path) = &args.output_path {
            std::fs::write(output_path, content.into_bytes())?;
        } else {
            println!("{}", content);
        }
        Ok(())
    };

    match args.output_format {
        OutputFormat::Text => {
            let content = format_text_output(&line_texts);
            write_output_str(content)?;
        }
        OutputFormat::Json => {
            let content = format_json_output(FormatJsonArgs {
                input_path: &args.image,
                input_hw: color_img.shape()[1..].try_into()?,
                word_rects: &word_rects,
            });
            write_output_str(content)?;
        }
        OutputFormat::Png => {
            let png_args = GeneratePngArgs {
                img: color_img.view(),
                line_rects: &line_rects,
                text_lines: &line_texts,
            };
            let annotated_img = generate_annotated_png(png_args);
            let Some(output_path) = args.output_path else {
                return Err("Output path must be specified when generating annotated PNG".into());
            };
            write_image(&output_path, annotated_img.view())?;
        }
    }

    if args.debug {
        println!(
            "Found {} words, {} lines in image of size {}x{}",
            word_rects.len(),
            line_rects.len(),
            color_img.size(2),
            color_img.size(1),
        );
    }

    Ok(())
}
