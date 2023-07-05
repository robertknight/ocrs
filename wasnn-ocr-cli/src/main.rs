use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::BufWriter;
use std::iter::zip;

use wasnn_imageproc::{draw_polygon, Point};
use wasnn_ocr::{OcrEngine, OcrEngineParams, TextItem};
use wasnn_tensor::{NdTensorViewMut, Tensor, TensorLayout, TensorView};

mod models;
use models::{load_model, ModelSource};

/// Read an image from `path` into a CHW tensor.
fn read_image(path: &str) -> Result<Tensor<f32>, Box<dyn Error>> {
    let input_img = image::open(path)?;
    let input_img = input_img.into_rgb8();

    let (width, height) = input_img.dimensions();

    let in_chans = 3;
    let mut float_img = Tensor::zeros(&[in_chans, height as usize, width as usize]);
    for c in 0..in_chans {
        let mut chan_img = float_img.nd_slice_mut([c]);
        for y in 0..height {
            for x in 0..width {
                chan_img[[y as usize, x as usize]] = input_img.get_pixel(x, y)[c] as f32 / 255.0
            }
        }
    }
    Ok(float_img)
}

/// Write a CHW image to a PNG file in `path`.
fn write_image(path: &str, img: TensorView<f32>) -> Result<(), Box<dyn Error>> {
    if img.ndim() != 3 {
        return Err("Expected CHW input".into());
    }

    let img_width = img.size(img.ndim() - 1);
    let img_height = img.size(img.ndim() - 2);
    let color_type = match img.size(img.ndim() - 3) {
        1 => png::ColorType::Grayscale,
        3 => png::ColorType::Rgb,
        4 => png::ColorType::Rgba,
        _ => return Err("Unsupported channel count".into()),
    };

    let mut hwc_img = img.to_owned();
    hwc_img.permute(&[1, 2, 0]); // CHW => HWC

    let out_img = image_from_tensor(hwc_img);
    let file = fs::File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, img_width as u32, img_height as u32);
    encoder.set_color(color_type);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&out_img)?;

    Ok(())
}

/// Convert an NCHW float tensor with values in the range [0, 1] to Vec<u8>
/// with values scaled to [0, 255].
fn image_from_tensor(tensor: TensorView<f32>) -> Vec<u8> {
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
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut values = VecDeque::new();
    let mut debug = false;
    let mut detection_model = None;
    let mut recognition_model = None;

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Value(val) => values.push_back(val.string()?),
            Long("debug") => {
                debug = true;
            }
            Long("detect-model") => {
                detection_model = Some(parser.value()?.string()?);
            }
            Long("rec-model") => {
                recognition_model = Some(parser.value()?.string()?);
            }
            Long("help") => {
                println!(
                    "Read text in images.

Usage: {bin_name} [OPTIONS] <image>

Options:

  --debug

    Enable debug output.

  --detect-model <path>

    Use a custom text detection model.

  --rec-model <path>

    Use a custom text recognition model.
",
                    bin_name = parser.bin_name().unwrap_or("wasnn-ocr")
                );
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    Ok(Args {
        detection_model,
        recognition_model,
        image: values.pop_front().ok_or("missing `<image>` arg")?,
        debug,
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

/// Utility for drawing into an image tensor.
struct Painter<'a, T> {
    /// CHW image tensor.
    surface: NdTensorViewMut<'a, T, 3>,

    /// Stroke color for RGB channels.
    stroke: [T; 3],
}

impl<'a, T: Copy + Default> Painter<'a, T> {
    /// Create a Painter which draws into the CHW tensor `surface`.
    fn new(surface: NdTensorViewMut<'a, T, 3>) -> Painter<'a, T> {
        Painter {
            surface,
            stroke: [T::default(); 3],
        }
    }

    /// Set the RGB color values used by the `draw_*` methods.
    fn set_stroke(&mut self, stroke: [T; 3]) {
        self.stroke = stroke;
    }

    /// Draw a polygon into the surface.
    fn draw_polygon(&mut self, points: &[Point]) {
        for i in 0..3 {
            draw_polygon(self.surface.slice_mut([i]), points, self.stroke[i]);
        }
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
    });

    let ocr_input = engine.prepare_input(color_img.view())?;
    let word_rects = engine.detect_words(&ocr_input)?;
    let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
    let line_texts = engine.recognize_text(&ocr_input, &line_rects)?;
    for line in line_texts.iter().flatten() {
        println!("{}", line);
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
        let mut painter = Painter::new(annotated_img.nd_view_mut());
        let colors = [[0.9, 0., 0.], [0., 0.9, 0.], [0., 0., 0.9]];

        for (line, color) in zip(line_texts.into_iter().flatten(), colors.into_iter().cycle()) {
            for text_word in line.words() {
                painter.set_stroke(color);
                painter.draw_polygon(&text_word.rotated_rect().corners());
            }
        }

        // Write out the annotated input image.
        write_image("ocr-debug-output.png", annotated_img.view())?;
    }

    Ok(())
}
