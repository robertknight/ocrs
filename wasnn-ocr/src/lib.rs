use std::collections::HashMap;
use std::error::Error;

use rayon::prelude::*;

use wasnn::ctc::{CtcDecoder, CtcHypothesis};
use wasnn::ops::{pad, resize, CoordTransformMode, NearestMode, ResizeMode, ResizeTarget};
use wasnn::{Dimension, Model, RunOptions};
use wasnn_imageproc::{bounding_rect, BoundingRect, Point, Polygon, Rect, RotatedRect};
use wasnn_tensor::{tensor, Tensor, TensorLayout, TensorView};

pub mod page_layout;

use page_layout::{find_connected_component_rects, find_text_lines, line_polygon};

/// Return the smallest multiple of `factor` that is >= `val`.
fn round_up<
    T: Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Rem<T, Output = T>,
>(
    val: T,
    factor: T,
) -> T {
    let rem = val % factor;
    (val + factor) - rem
}

// nb. The "E" before "ABCDE" should be the EUR symbol.
const DEFAULT_ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~EABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// The value used to represent fully black pixels in OCR input images
/// prepared by [prepare_image].
pub const BLACK_VALUE: f32 = -0.5;

/// Convert a CHW image into a greyscale image.
///
/// This function is intended to approximately match torchvision's RGB =>
/// greyscale conversion when using `torchvision.io.read_image(path,
/// ImageReadMode.GRAY)`, which is used when training models with greyscale
/// inputs. torchvision internally uses libpng's `png_set_rgb_to_gray`.
///
/// `normalize_pixel` is a function applied to each greyscale pixel value before
/// it is written into the output tensor.
fn greyscale_image<F: Fn(f32) -> f32>(img: TensorView<f32>, normalize_pixel: F) -> Tensor<f32> {
    let [chans, height, width]: [usize; 3] = img.shape().try_into().expect("expected 3 dim input");
    assert!(
        chans == 1 || chans == 3 || chans == 4,
        "expected greyscale, RGB or RGBA input image"
    );

    let mut output = Tensor::zeros(&[1, height, width]);

    let used_chans = chans.min(3); // For RGBA images, only RGB channels are used
    let chan_weights: &[f32] = if chans == 1 {
        &[1.]
    } else {
        // ITU BT.601 weights for RGB => luminance conversion. These match what
        // torchvision uses. See also https://stackoverflow.com/a/596241/434243.
        &[0.299, 0.587, 0.114]
    };

    let img = img.nd_view();
    let mut out_lum_chan = output.nd_slice_mut([0]);

    for y in 0..height {
        for x in 0..width {
            let mut pixel = 0.;
            for c in 0..used_chans {
                pixel += img[[c, y, x]] * chan_weights[c];
            }
            out_lum_chan[[y, x]] = normalize_pixel(pixel);
        }
    }
    output
}

/// Prepare an image for use with [detect_words] and [recognize_text_lines].
///
/// This converts an input CHW image with values in the range 0-1 to a greyscale
/// image with values in the range `ZERO_VALUE` to `ZERO_VALUE + 1`.
fn prepare_image(image: TensorView<f32>) -> Tensor<f32> {
    greyscale_image(image, |pixel| pixel + BLACK_VALUE)
}

/// Detect text words in a greyscale image.
///
/// `image` is a greyscale CHW image with values in the range `ZERO_VALUE` to
/// `ZERO_VALUE + 1`. `model` is a model which takes an NCHW input tensor and
/// returns a binary segmentation mask predicting whether each pixel is part of
/// a text word or not. The image is padded and resized to the model's expected
/// input size before performing detection.
///
/// The result is an unsorted list of the oriented bounding rectangles of
/// connected components (ie. text words) in the mask.
fn detect_words(
    mut image: TensorView<f32>,
    model: &Model,
    debug: bool,
) -> Result<Vec<RotatedRect>, Box<dyn Error>> {
    let input_id = model
        .input_ids()
        .first()
        .copied()
        .expect("model has no inputs");
    let input_shape = model
        .node_info(input_id)
        .and_then(|info| info.shape())
        .ok_or("model does not specify expected input shape")?;
    let output_id = model
        .output_ids()
        .first()
        .copied()
        .expect("model has no outputs");

    let [_, img_height, img_width] = image.dims();
    image.insert_dim(0); // Add batch dimension

    let bilinear_resize = |img: TensorView, height, width| {
        resize(
            img,
            ResizeTarget::Sizes(&tensor!([1, 1, height, width])),
            ResizeMode::Linear,
            CoordTransformMode::default(),
            NearestMode::default(),
        )
    };

    let (in_height, in_width) = match input_shape[..] {
        [_, _, Dimension::Fixed(h), Dimension::Fixed(w)] => (h, w),
        _ => {
            return Err("failed to get model dims".into());
        }
    };

    // Pad small images to the input size of the text detection model. This is
    // needed because simply scaling small images up to a fixed size may produce
    // very large or distorted text that is hard for detection/recognition to
    // process.
    //
    // Padding images is however inefficient because it means that we are
    // potentially feeding a lot of blank pixels into the text detection model.
    // It would be better if text detection were able to accept variable-sized
    // inputs, within some limits.
    let pad_bottom = (in_height as i32 - img_height as i32).max(0);
    let pad_right = (in_width as i32 - img_width as i32).max(0);
    let grey_img = if pad_bottom > 0 || pad_right > 0 {
        pad(
            image.view(),
            &tensor!([0, 0, 0, 0, 0, 0, pad_bottom, pad_right]),
            BLACK_VALUE,
        )?
    } else {
        image.to_tensor()
    };

    // Resize images to the text detection model's input size.
    let resized_grey_img = bilinear_resize(grey_img.view(), in_height as i32, in_width as i32)?;

    // Run text detection model to compute a probability mask indicating whether
    // each pixel is part of a text word or not.
    let outputs = model.run(
        &[(input_id, (&resized_grey_img).into())],
        &[output_id],
        if debug {
            Some(RunOptions {
                timing: true,
                verbose: false,
            })
        } else {
            None
        },
    )?;

    // Resize probability mask to original input size and apply threshold to get a
    // binary text/not-text mask.
    let text_mask = &outputs[0].as_float_ref().unwrap();
    let text_mask = bilinear_resize(
        text_mask.slice((
            ..,
            ..,
            ..(in_height - pad_bottom as usize),
            ..(in_width - pad_right as usize),
        )),
        img_height as i32,
        img_width as i32,
    )?;
    let threshold = 0.2;
    let binary_mask = text_mask.map(|prob| if prob > threshold { 1i32 } else { 0 });

    // Distance to expand bounding boxes by. This is useful when the model is
    // trained to assign a positive label to pixels in a smaller area than the
    // ground truth, which may be done to create separation between adjacent
    // objects.
    let expand_dist = 3.;

    let word_rects = find_connected_component_rects(binary_mask.nd_slice([0, 0]), expand_dist);

    Ok(word_rects)
}

/// Details about a text line needed to prepare the input to the text
/// recognition model.
#[derive(Clone)]
struct TextRecLine {
    /// Index of this line in the list of lines found in the image.
    index: usize,

    /// Region of the image containing this line.
    region: Polygon,

    /// Width to resize this line to.
    resized_width: i32,
}

/// Prepare an NCHW tensor containing a batch of text line images, for input
/// into the text recognition model.
///
/// For each line in `lines`, the line region is extracted from `image`, resized
/// to a fixed `output_height` and a line-specific width, then copied to the
/// output tensor. Lines in the batch can have different widths, so the output
/// is padded on the right side to a common width of `output_width`.
fn prepare_text_line_batch(
    image: &TensorView<f32>,
    lines: &[TextRecLine],
    page_rect: Rect,
    output_height: usize,
    output_width: usize,
) -> Tensor<f32> {
    let mut output = Tensor::zeros(&[lines.len(), 1, output_height, output_width]);
    output.data_mut().fill(BLACK_VALUE);

    // Page rect adjusted to only contain coordinates that are valid for
    // indexing into the input image.
    let page_index_rect = page_rect.adjust_tlbr(0, 0, -1, -1);

    for (group_line_index, line) in lines.iter().enumerate() {
        let grey_chan = image.nd_slice([0]);

        let line_rect = line.region.bounding_rect();
        let mut line_img = Tensor::zeros(&[
            1,
            1,
            line_rect.height() as usize,
            line_rect.width() as usize,
        ]);
        line_img.data_mut().fill(BLACK_VALUE);

        let mut line_img_chan = line_img.nd_slice_mut([0, 0]);
        for in_p in line.region.fill_iter() {
            let out_p = Point::from_yx(in_p.y - line_rect.top(), in_p.x - line_rect.left());
            if !page_index_rect.contains_point(in_p) || !page_index_rect.contains_point(out_p) {
                continue;
            }
            line_img_chan[[out_p.y as usize, out_p.x as usize]] =
                grey_chan[[in_p.y as usize, in_p.x as usize]];
        }

        let resized_line_img = resize(
            line_img.view(),
            ResizeTarget::Sizes(&tensor!([1, 1, output_height as i32, line.resized_width])),
            ResizeMode::Linear,
            CoordTransformMode::default(),
            NearestMode::default(),
        )
        .unwrap();

        output
            .slice_mut((group_line_index, 0, .., ..(line.resized_width as usize)))
            .copy_from(&resized_line_img.squeezed());
    }

    output
}

/// Recognize text lines in an image.
///
/// `image` is a CHW greyscale image with values in the range `ZERO_VALUE` to
/// `ZERO_VALUE + 1`. `lines` is a list of detected text lines, where each line
/// is a sequence of word rects. `model` is a recognition model which accepts an
/// NCHW tensor of greyscale line images and outputs a `[sequence, batch, label]`
/// tensor of log probabilities of character classes, which must be converted to
/// a character sequence using CTC decoding.
///
/// The result is a list containing the recognized text for each input line.
fn recognize_text_lines(
    image: TensorView<f32>,
    lines: &[Vec<RotatedRect>],
    model: &Model,
    debug: bool,
) -> Result<Vec<String>, Box<dyn Error>> {
    let [_, img_height, img_width] = image.dims();
    let page_rect = Rect::from_hw(img_height as i32, img_width as i32);
    let rec_input_id = model
        .input_ids()
        .first()
        .copied()
        .expect("recognition model has no inputs");
    let rec_input_shape = model
        .node_info(rec_input_id)
        .and_then(|info| info.shape())
        .ok_or("recognition model does not specify input shape")?;
    let rec_output_id = model
        .output_ids()
        .first()
        .copied()
        .ok_or("recognition model has no outputs")?;

    // Get the input height that the recognition model expects.
    let rec_img_height = match rec_input_shape[2] {
        Dimension::Fixed(size) => size,
        Dimension::Symbolic(_) => 50,
    };

    // Compute width to resize a text line image to, for a given height.
    fn resized_line_width(orig_width: i32, orig_height: i32, height: i32) -> i32 {
        // Min/max widths for resized line images. These must match the PyTorch
        // `HierTextRecognition` dataset loader.
        let min_width = 10.;
        let max_width = 800.;
        let aspect_ratio = orig_width as f32 / orig_height as f32;
        (height as f32 * aspect_ratio).max(min_width).min(max_width) as i32
    }

    // Group lines into batches which will have similar widths after resizing
    // to a fixed height.
    //
    // It is more efficient to run recognition on multiple lines at once, but
    // all line images in a batch must be padded to an equal length. Some
    // computation is wasted on shorter lines in the batch. Choosing batches
    // such that all line images have a similar width reduces this wastage.
    // There is a trade-off between maximizing the batch size and minimizing
    // the variance in width of images in the batch.
    let mut line_groups: HashMap<i32, Vec<TextRecLine>> = HashMap::new();
    for (line_index, word_rects) in lines.iter().enumerate() {
        let line_rect = bounding_rect(word_rects).expect("line has no words");
        let resized_width =
            resized_line_width(line_rect.width(), line_rect.height(), rec_img_height as i32);
        let group_width = round_up(resized_width, 50);
        line_groups
            .entry(group_width)
            .or_default()
            .push(TextRecLine {
                index: line_index,
                region: Polygon::new(line_polygon(word_rects)),
                resized_width,
            });
    }

    // Split large line groups up to better exploit parallelism in the loop
    // below.
    let max_lines_per_group = 20;
    let line_groups: Vec<(i32, Vec<TextRecLine>)> = line_groups
        .into_iter()
        .flat_map(|(group_width, lines)| {
            lines
                .chunks(max_lines_per_group)
                .map(|chunk| (group_width, chunk.to_vec()))
                .collect::<Vec<_>>()
        })
        .collect();

    // Run text recognition on batches of lines. Produces a map of line index
    // to recognition result.
    let line_rec_results: HashMap<usize, CtcHypothesis> = line_groups
        .par_iter()
        .flat_map(|(group_width, lines)| {
            if debug {
                println!(
                    "Processing group of {} lines of width {}",
                    lines.len(),
                    group_width,
                );
            }

            let rec_input = prepare_text_line_batch(
                &image,
                lines,
                page_rect,
                rec_img_height,
                *group_width as usize,
            );

            // Perform text recognition on the line batch.
            let rec_output = model
                .run(
                    &[(rec_input_id, (&rec_input).into())],
                    &[rec_output_id],
                    None,
                )
                .unwrap();
            let mut rec_sequence = rec_output[0].as_float_ref().unwrap().to_tensor();

            // Transpose from [seq, batch, class] => [batch, seq, class]
            rec_sequence.permute(&[1, 0, 2]);

            lines
                .iter()
                .enumerate()
                .map(|(group_line_index, line)| {
                    let decoder = CtcDecoder::new();
                    let input_seq = rec_sequence.slice([group_line_index]);
                    let decode_result = decoder.decode_greedy(input_seq.clone());
                    (line.index, decode_result)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let line_texts = (0..lines.len())
        .filter_map(|idx| line_rec_results.get(&idx))
        .map(|result| result.to_string(DEFAULT_ALPHABET))
        .collect();

    Ok(line_texts)
}

/// Configuration for an [OcrEngine] instance.
#[derive(Default)]
pub struct OcrEngineParams {
    /// Model used to detect text words in the image.
    pub detection_model: Option<Model>,

    /// Model used to recognize lines of text in the image.
    pub recognition_model: Option<Model>,

    /// Enable debug logging.
    pub debug: bool,
}

/// Detects and recognizes text in images.
///
/// OcrEngine uses machine learning models to detect text, analyze layout
/// and recognize text in an image.
pub struct OcrEngine {
    params: OcrEngineParams,
}

/// Input image for OCR analysis. Instances are created using
/// [OcrEngine::prepare_input]
pub struct OcrInput {
    image: Tensor<f32>,
}

impl OcrEngine {
    /// Construct a new engine from a given configuration.
    pub fn new(params: OcrEngineParams) -> OcrEngine {
        OcrEngine { params }
    }

    /// Preprocess an image for use with other methods of the engine.
    pub fn prepare_input(&self, image: TensorView<f32>) -> Result<OcrInput, Box<dyn Error>> {
        Ok(OcrInput {
            image: prepare_image(image),
        })
    }

    /// Detect text words in an image.
    ///
    /// Returns an unordered list of the oriented bounding rectangles of each
    /// word found.
    pub fn detect_words(&self, input: &OcrInput) -> Result<Vec<RotatedRect>, Box<dyn Error>> {
        let Some(detection_model) = self.params.detection_model.as_ref() else {
            return Err("Detection model not loaded".into());
        };
        detect_words(input.image.view(), detection_model, self.params.debug)
    }

    /// Perform layout analysis to group words into lines and sort them in
    /// reading order.
    ///
    /// `words` is an unordered list of text word rectangles found by
    /// [OcrEngine::detect_words]. The result is a list of lines, in reading
    /// order. Each line is a sequence of word bounding rectangles, in reading
    /// order.
    pub fn find_text_lines(
        &self,
        input: &OcrInput,
        words: &[RotatedRect],
    ) -> Vec<Vec<RotatedRect>> {
        let [_, img_height, img_width] = input.image.dims();
        let page_rect = Rect::from_hw(img_height as i32, img_width as i32);
        find_text_lines(words, page_rect)
    }

    /// Recognize lines of text in an image.
    ///
    /// `lines` is an ordered list of the text line boxes in an image,
    /// produced by [OcrEngine::find_text_lines].
    pub fn recognize_text(
        &self,
        input: &OcrInput,
        lines: &[Vec<RotatedRect>],
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let Some(recognition_model) = self.params.recognition_model.as_ref() else {
            return Err("Recognition model not loaded".into());
        };
        recognize_text_lines(
            input.image.view(),
            lines,
            recognition_model,
            self.params.debug,
        )
    }

    /// Convenience API that extracts all text from an image as a single string.
    pub fn get_text(&self, input: &OcrInput) -> Result<String, Box<dyn Error>> {
        let word_rects = self.detect_words(input)?;
        let line_rects = self.find_text_lines(input, &word_rects);
        let line_texts = self.recognize_text(input, &line_rects)?;
        Ok(line_texts.join("\n"))
    }
}
