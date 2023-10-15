use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::BufWriter;
use std::iter::zip;

use serde_json::json;

use wasnn_imageproc::{min_area_rect, BoundingRect, Painter, Point, PointF, Rgb, RotatedRect};
use wasnn_ocr::{DecodeMethod, OcrEngine, OcrEngineParams, TextItem};
use wasnn_tensor::{Layout, NdTensor, NdTensorView};

mod models;
use models::{load_model, ModelSource};

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

/// Generate a JSON representation of detected word boxes in an image.
///
/// The format here matches that used by the layout model / evaluation tools.
fn word_boxes_json(
    img_path: &str,
    image: NdTensorView<f32, 3>,
    boxes: &[RotatedRect],
) -> serde_json::Value {
    let word_items: Vec<_> = boxes
        .iter()
        .map(|wr| {
            let br = wr.bounding_rect();
            let coords = [br.left(), br.top(), br.right(), br.bottom()].map(|c| c.round() as i32);
            json!({
                "coords": coords,
            })
        })
        .collect();

    json!({
        "url": img_path,
        "resolution": {
            "width": image.size(2),
            "height": image.size(1),
        },

        // nb. Since we haven't got layout analysis info here, we just put all
        // the words in one paragraph.
        "paragraphs": [{
            "words": serde_json::Value::Array(word_items),
        }]
    })
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

    /// Export word boxes from text detection to a JSON file.
    export_boxes: Option<String>,

    /// Use beam search for sequence decoding.
    beam_search: bool,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut beam_search = false;
    let mut debug = false;
    let mut detection_model = None;
    let mut export_boxes = None;
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
            Long("export-boxes") => {
                export_boxes = Some(parser.value()?.string()?);
            }
            Long("rec-model") => {
                recognition_model = Some(parser.value()?.string()?);
            }
            Long("help") => {
                println!(
                    "Read text in images.

Usage: {bin_name} [OPTIONS] <image>

Options:

  --beam

    Use beam search for decoding.

  --debug

    Enable debug output.

  --detect-model <path>

    Use a custom text detection model.

  --rec-model <path>

    Use a custom text recognition model.

  --export-boxes <path>

    Export detected word boxes in JSON format.
",
                    bin_name = parser.bin_name().unwrap_or("wasnn-ocr")
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
        export_boxes,
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
const DETECTION_MODEL: &str = "http://localhost:2000/text-detection.model";

/// Default text recognition model.
const RECOGNITION_MODEL: &str = "http://localhost:2000/text-recognition.model";

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
    for line in line_texts.iter().flatten() {
        println!("{}", line);
    }

    if let Some(boxes_path) = args.export_boxes {
        let json_data = word_boxes_json(&args.image, color_img.view(), &word_rects);
        let json_bytes = serde_json::to_vec_pretty(&json_data)?;
        std::fs::write(boxes_path, &json_bytes)?;
    }

    if args.debug {
        println!(
            "Found {} words, {} lines in image of size {}x{}",
            word_rects.len(),
            line_rects.len(),
            color_img.size(2),
            color_img.size(1),
        );

        let mut annotated_img = color_img;
        let mut painter = Painter::new(annotated_img.view_mut());

        // Colors chosen from https://www.w3.org/wiki/CSS/Properties/color/keywords.
        //
        // Light colors for text detection outputs, darker colors for
        // corresponding text recognition outputs.
        const CORAL: Rgb = [255, 127, 80];
        const DARKSEAGREEN: Rgb = [143, 188, 143];
        const CORNFLOWERBLUE: Rgb = [100, 149, 237];

        const CRIMSON: Rgb = [220, 20, 60];
        const DARKGREEN: Rgb = [0, 100, 0];
        const DARKBLUE: Rgb = [0, 0, 139];

        const LIGHT_GRAY: Rgb = [200, 200, 200];

        let u8_to_f32 = |x: u8| x as f32 / 255.;
        let floor_point = |p: PointF| Point::from_yx(p.y as i32, p.x as i32);

        // Draw line bounding rects from layout analysis step.
        for line in line_rects.iter() {
            let line_points: Vec<_> = line
                .iter()
                .flat_map(|word_rect| word_rect.corners().into_iter())
                .collect();
            if let Some(line_rect) = min_area_rect(&line_points) {
                painter.set_stroke(LIGHT_GRAY.map(u8_to_f32));
                painter.draw_polygon(&line_rect.corners().map(floor_point));
            };
        }

        // Draw word bounding rects from text detection step, grouped by line.
        let colors = [CORAL, DARKSEAGREEN, CORNFLOWERBLUE];
        for (line, color) in zip(line_rects.iter(), colors.into_iter().cycle()) {
            for word_rect in line {
                painter.set_stroke(color.map(u8_to_f32));
                painter.draw_polygon(&word_rect.corners().map(floor_point));
            }
        }

        // Draw word bounding rects from text recognition step. These may be
        // different as they are computed from the bounding boxes of recognized
        // characters.
        let colors = [CRIMSON, DARKGREEN, DARKBLUE];
        for (line, color) in zip(line_texts.into_iter(), colors.into_iter().cycle()) {
            let Some(line) = line else {
                // Skip lines where recognition produced no output.
                continue;
            };
            for text_word in line.words() {
                painter.set_stroke(color.map(u8_to_f32));
                painter.draw_polygon(&text_word.rotated_rect().corners().map(floor_point));
            }
        }

        // Write out the annotated input image.
        write_image("ocr-debug-output.png", annotated_img.view())?;
    }

    Ok(())
}
