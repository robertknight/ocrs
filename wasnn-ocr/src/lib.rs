use std::collections::HashMap;
use std::error::Error;

use rayon::prelude::*;

use wasnn::ctc::{CtcDecoder, CtcHypothesis};
use wasnn::ops::{pad, resize, CoordTransformMode, NearestMode, ResizeMode, ResizeTarget};
use wasnn::{Dimension, Model, RunOptions};
use wasnn_imageproc::{bounding_rect, BoundingRect, Line, Point, Polygon, Rect, RotatedRect};
use wasnn_tensor::{Layout, NdTensor, NdTensorView, NdView, Tensor, TensorView, View};

mod log;
pub mod page_layout;
mod text_items;
mod wasm_api;

use page_layout::{find_connected_component_rects, find_text_lines, line_polygon};
pub use text_items::{TextChar, TextItem, TextLine, TextWord};

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
fn greyscale_image<F: Fn(f32) -> f32>(
    img: NdTensorView<f32, 3>,
    normalize_pixel: F,
) -> NdTensor<f32, 3> {
    let [chans, height, width] = img.shape();
    assert!(
        chans == 1 || chans == 3 || chans == 4,
        "expected greyscale, RGB or RGBA input image"
    );

    let mut output = NdTensor::zeros([1, height, width]);

    let used_chans = chans.min(3); // For RGBA images, only RGB channels are used
    let chan_weights: &[f32] = if chans == 1 {
        &[1.]
    } else {
        // ITU BT.601 weights for RGB => luminance conversion. These match what
        // torchvision uses. See also https://stackoverflow.com/a/596241/434243.
        &[0.299, 0.587, 0.114]
    };

    let mut out_lum_chan = output.slice_mut([0]);

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
/// image with values in the range `BLACK_VALUE` to `BLACK_VALUE + 1`.
fn prepare_image(image: NdTensorView<f32, 3>) -> NdTensor<f32, 3> {
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
    image: NdTensorView<f32, 3>,
    model: &Model,
    debug: bool,
) -> Result<Vec<RotatedRect>, Box<dyn Error>> {
    let input_id = model
        .input_ids()
        .first()
        .copied()
        .ok_or("model has no inputs")?;
    let input_shape = model
        .node_info(input_id)
        .and_then(|info| info.shape())
        .ok_or("model does not specify expected input shape")?;

    let [img_chans, img_height, img_width] = image.shape();

    // Add batch dim
    let image = image.reshaped([1, img_chans, img_height, img_width]);

    let bilinear_resize = |img: TensorView, height, width| {
        let sizes = &[1, 1, height, width];
        resize(
            img,
            ResizeTarget::Sizes(sizes.into()),
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
        let pads = &[0, 0, 0, 0, 0, 0, pad_bottom, pad_right];
        pad(image.as_dyn(), &pads.into(), BLACK_VALUE)?
    } else {
        image.as_dyn().to_tensor()
    };

    // Resize images to the text detection model's input size.
    let resized_grey_img = bilinear_resize(grey_img.view(), in_height as i32, in_width as i32)?;

    // Run text detection model to compute a probability mask indicating whether
    // each pixel is part of a text word or not.
    let text_mask: Tensor<f32> = model
        .run_one(
            (&resized_grey_img).into(),
            if debug {
                Some(RunOptions {
                    timing: true,
                    verbose: false,
                })
            } else {
                None
            },
        )?
        .try_into()?;

    // Resize probability mask to original input size and apply threshold to get a
    // binary text/not-text mask.
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
    let binary_mask = text_mask.map(|prob| if *prob > threshold { 1i32 } else { 0 });

    // Distance to expand bounding boxes by. This is useful when the model is
    // trained to assign a positive label to pixels in a smaller area than the
    // ground truth, which may be done to create separation between adjacent
    // objects.
    let expand_dist = 3.;

    let word_rects =
        find_connected_component_rects(binary_mask.slice([0, 0]).nd_view(), expand_dist);

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
    resized_width: u32,
}

/// Prepare an NCHW tensor containing a batch of text line images, for input
/// into the text recognition model.
///
/// For each line in `lines`, the line region is extracted from `image`, resized
/// to a fixed `output_height` and a line-specific width, then copied to the
/// output tensor. Lines in the batch can have different widths, so the output
/// is padded on the right side to a common width of `output_width`.
fn prepare_text_line_batch(
    image: &NdTensorView<f32, 3>,
    lines: &[TextRecLine],
    page_rect: Rect,
    output_height: usize,
    output_width: usize,
) -> NdTensor<f32, 4> {
    let mut output = NdTensor::zeros([lines.len(), 1, output_height, output_width]);
    output.data_mut().fill(BLACK_VALUE);

    // Page rect adjusted to only contain coordinates that are valid for
    // indexing into the input image.
    let page_index_rect = page_rect.adjust_tlbr(0, 0, -1, -1);

    for (group_line_index, line) in lines.iter().enumerate() {
        let grey_chan = image.slice([0]);

        let line_rect = line.region.bounding_rect();
        let mut line_img =
            NdTensor::zeros([line_rect.height() as usize, line_rect.width() as usize]);
        line_img.data_mut().fill(BLACK_VALUE);

        for in_p in line.region.fill_iter() {
            let out_p = Point::from_yx(in_p.y - line_rect.top(), in_p.x - line_rect.left());
            if !page_index_rect.contains_point(in_p) || !page_index_rect.contains_point(out_p) {
                continue;
            }
            line_img[[out_p.y as usize, out_p.x as usize]] =
                grey_chan[[in_p.y as usize, in_p.x as usize]];
        }

        let resized_shape = &[1, 1, output_height as i32, line.resized_width as i32];
        let resized_line_img = resize(
            line_img
                .reshaped([1, 1, line_img.size(0), line_img.size(1)])
                .as_dyn(),
            ResizeTarget::Sizes(resized_shape.into()),
            ResizeMode::Linear,
            CoordTransformMode::default(),
            NearestMode::default(),
        )
        .unwrap();

        let resized_line_img: NdTensorView<f32, 2> =
            resized_line_img.squeezed().try_into().unwrap();

        output
            .slice_mut((group_line_index, 0, .., ..(line.resized_width as usize)))
            .copy_from(&resized_line_img);
    }

    output
}

/// Return the bounding rectangle of the slice of a polygon with X coordinates
/// between `min_x` and `max_x` inclusive.
fn polygon_slice_bounding_rect(
    poly: Polygon<i32, &[Point]>,
    min_x: i32,
    max_x: i32,
) -> Option<Rect> {
    poly.edges()
        .filter_map(|e| {
            let e = e.rightwards();

            // Filter out edges that don't overlap [min_x, max_x].
            if (e.start.x < min_x && e.end.x < min_x) || (e.start.x > max_x && e.end.x > max_x) {
                return None;
            }

            // Truncate edge to [min_x, max_x].
            let trunc_edge_start = e
                .to_f32()
                .y_for_x(min_x as f32)
                .map(|y| Point::from_yx(y.round() as i32, min_x))
                .unwrap_or(e.start);

            let trunc_edge_end = e
                .to_f32()
                .y_for_x(max_x as f32)
                .map(|y| Point::from_yx(y.round() as i32, max_x))
                .unwrap_or(e.end);

            Some(Line::from_endpoints(trunc_edge_start, trunc_edge_end))
        })
        .fold(None, |bounding_rect, e| {
            let edge_br = e.bounding_rect();
            bounding_rect.map(|br| br.union(edge_br)).or(Some(edge_br))
        })
}

/// Method used to decode sequence model outputs to a sequence of labels.
///
/// See [CtcDecoder] for more details.
#[derive(Copy, Clone, Default)]
pub enum DecodeMethod {
    #[default]
    Greedy,
    BeamSearch {
        width: u32,
    },
}

#[derive(Clone, Default)]
pub struct RecognitionOpt {
    pub debug: bool,

    /// Method used to decode character sequence outputs to character values.
    pub decode_method: DecodeMethod,
}

/// Input and output from recognition for a single text line.
struct LineRecResult {
    /// Input to the recognition model.
    line: TextRecLine,

    /// Length of input sequences to recognition model, padded so that all
    /// lines in batch have the same length.
    rec_input_len: usize,

    /// Length of output sequences from recognition model, used as input to
    /// CTC decoding.
    ctc_input_len: usize,

    /// Output label sequence produced by CTC decoding.
    ctc_output: CtcHypothesis,
}

/// Combine information from the input and output of text line recognition
/// to produce [TextLine]s containing character sequences and bounding boxes
/// for each line.
///
/// Entries in the result may be `None` if no text was recognized for a line.
fn text_lines_from_recognition_results(results: &[LineRecResult]) -> Vec<Option<TextLine>> {
    results
        .iter()
        .map(|result| {
            let line_rect = result.line.region.bounding_rect();
            let x_scale_factor = (line_rect.width() as f32) / (result.line.resized_width as f32);

            // Calculate how much the recognition model downscales the image
            // width. We assume this will be an integer factor, or close to it
            // if the input width is not an exact multiple of the downscaling
            // factor.
            let downsample_factor =
                (result.rec_input_len as f32 / result.ctc_input_len as f32).round() as u32;

            let steps = result.ctc_output.steps();
            let text_line: Vec<TextChar> = steps
                .iter()
                .enumerate()
                .filter_map(|(i, step)| {
                    // X coord range of character in line recognition input image.
                    let start_x = step.pos * downsample_factor;
                    let end_x = if let Some(next_step) = steps.get(i + 1) {
                        next_step.pos * downsample_factor
                    } else {
                        result.line.resized_width
                    };

                    // Map X coords to those of the input image.
                    let [start_x, end_x] = [start_x, end_x]
                        .map(|x| line_rect.left() + (x as f32 * x_scale_factor) as i32);

                    // Since the recognition input is padded, it is possible to
                    // get predicted characters in the output with positions
                    // that correspond to the padding region, and thus are
                    // outside the bounds of the original line. Ignore these.
                    if start_x >= line_rect.right() {
                        return None;
                    }

                    let char = DEFAULT_ALPHABET
                        .chars()
                        .nth((step.label - 1) as usize)
                        .unwrap_or('?');

                    Some(TextChar {
                        char,
                        rect: polygon_slice_bounding_rect(
                            result.line.region.borrow(),
                            start_x,
                            end_x,
                        )
                        .expect("invalid X coords"),
                    })
                })
                .collect();

            if text_line.is_empty() {
                None
            } else {
                Some(TextLine::new(text_line))
            }
        })
        .collect()
}

/// Encapsulates validation and execution of the text line recognition model.
struct RecognitionModel {
    model: Model,
    input_id: usize,
    input_shape: Vec<Dimension>,
    output_id: usize,
}

impl RecognitionModel {
    /// Validate that a model has the expected inputs and outputs for
    /// a text recognition model and wrap it as a [RecognitionModel].
    fn from_model(model: Model) -> Result<RecognitionModel, Box<dyn Error>> {
        let input_id = model
            .input_ids()
            .first()
            .copied()
            .ok_or("recognition model has no inputs")?;
        let input_shape = model
            .node_info(input_id)
            .and_then(|info| info.shape())
            .ok_or("recognition model does not specify input shape")?;
        let output_id = model
            .output_ids()
            .first()
            .copied()
            .ok_or("recognition model has no outputs")?;
        Ok(RecognitionModel {
            model,
            input_id,
            input_shape: input_shape.into_iter().collect(),
            output_id,
        })
    }

    /// Return the expected height of input line images.
    fn input_height(&self) -> u32 {
        match self.input_shape[2] {
            Dimension::Fixed(size) => size.try_into().unwrap(),
            Dimension::Symbolic(_) => 50,
        }
    }

    /// Run text recognition on an NCHW batch of text line images, and return
    /// a `[batch, seq, label]` tensor of class probabilities.
    fn run(&self, input: NdTensor<f32, 4>) -> Result<NdTensor<f32, 3>, Box<dyn Error>> {
        let input: Tensor<f32> = input.into();
        let mut rec_output =
            self.model
                .run(&[(self.input_id, (&input).into())], &[self.output_id], None)?;
        let mut rec_sequence: Tensor<f32> = rec_output.remove(0).try_into()?;

        // Transpose from [seq, batch, class] => [batch, seq, class]
        rec_sequence.permute(&[1, 0, 2]);

        Ok(rec_sequence.nd_view().to_tensor())
    }
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
/// Entries in the result can be `None` if no text was found in a line.
fn recognize_text_lines(
    image: NdTensorView<f32, 3>,
    lines: &[Vec<RotatedRect>],
    model: &RecognitionModel,
    opts: RecognitionOpt,
) -> Result<Vec<Option<TextLine>>, Box<dyn Error>> {
    let RecognitionOpt {
        debug,
        decode_method,
    } = opts;

    let [_, img_height, img_width] = image.shape();
    let page_rect = Rect::from_hw(img_height as i32, img_width as i32);

    // Compute width to resize a text line image to, for a given height.
    fn resized_line_width(orig_width: i32, orig_height: i32, height: i32) -> u32 {
        // Min/max widths for resized line images. These must match the PyTorch
        // `HierTextRecognition` dataset loader.
        let min_width = 10.;
        let max_width = 800.;
        let aspect_ratio = orig_width as f32 / orig_height as f32;
        (height as f32 * aspect_ratio).max(min_width).min(max_width) as u32
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
    let rec_img_height = model.input_height();
    let mut line_groups: HashMap<i32, Vec<TextRecLine>> = HashMap::new();
    for (line_index, word_rects) in lines.iter().enumerate() {
        let line_rect = bounding_rect(word_rects.iter())
            .expect("line has no words")
            .integral_bounding_rect();
        let resized_width =
            resized_line_width(line_rect.width(), line_rect.height(), rec_img_height as i32);
        let group_width = round_up(resized_width, 50);
        line_groups
            .entry(group_width as i32)
            .or_default()
            .push(TextRecLine {
                index: line_index,
                region: Polygon::new(line_polygon(word_rects)),
                resized_width,
            });
    }

    // Split large line groups up into smaller batches that can be processed
    // in parallel.
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

    // Run text recognition on batches of lines.
    let mut line_rec_results: Vec<LineRecResult> = line_groups
        .into_par_iter()
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
                &lines,
                page_rect,
                rec_img_height as usize,
                group_width as usize,
            );

            // TODO - Propagate errors from recognition model to caller.
            let rec_output = model.run(rec_input).expect("recognition failed");
            let ctc_input_len = rec_output.shape()[1];

            // Apply CTC decoding to get the label sequence for each line.
            lines
                .into_iter()
                .enumerate()
                .map(|(group_line_index, line)| {
                    let decoder = CtcDecoder::new();
                    let input_seq = rec_output.slice([group_line_index]);
                    let ctc_output = match decode_method {
                        DecodeMethod::Greedy => decoder.decode_greedy(input_seq),
                        DecodeMethod::BeamSearch { width } => decoder.decode_beam(input_seq, width),
                    };
                    LineRecResult {
                        line,
                        rec_input_len: group_width as usize,
                        ctc_input_len,
                        ctc_output,
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // The recognition outputs are in a different order than the inputs due to
    // batching and parallel processing. Re-sort them into input order.
    line_rec_results.sort_by_key(|result| result.line.index);

    let text_lines = text_lines_from_recognition_results(&line_rec_results);

    Ok(text_lines)
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

    pub decode_method: DecodeMethod,
}

/// Detects and recognizes text in images.
///
/// OcrEngine uses machine learning models to detect text, analyze layout
/// and recognize text in an image.
pub struct OcrEngine {
    detection_model: Option<Model>,
    recognition_model: Option<RecognitionModel>,
    debug: bool,
    decode_method: DecodeMethod,
}

/// Input image for OCR analysis. Instances are created using
/// [OcrEngine::prepare_input]
pub struct OcrInput {
    /// CHW tensor with normalized pixel values in [BLACK_VALUE, BLACK_VALUE + 1.].
    pub(crate) image: NdTensor<f32, 3>,
}

impl OcrEngine {
    /// Construct a new engine from a given configuration.
    pub fn new(params: OcrEngineParams) -> Result<OcrEngine, Box<dyn Error>> {
        let recognition_model = params
            .recognition_model
            .map(RecognitionModel::from_model)
            .transpose()?;
        Ok(OcrEngine {
            detection_model: params.detection_model,
            recognition_model,
            debug: params.debug,
            decode_method: params.decode_method,
        })
    }

    /// Preprocess an image for use with other methods of the engine.
    ///
    /// The input `image` should be a CHW tensor with values in the range 0-1
    /// and either 1 (grey), 3 (RGB) or 4 (RGBA) channels.
    pub fn prepare_input(&self, image: NdTensorView<f32, 3>) -> Result<OcrInput, Box<dyn Error>> {
        Ok(OcrInput {
            image: prepare_image(image),
        })
    }

    /// Detect text words in an image.
    ///
    /// Returns an unordered list of the oriented bounding rectangles of each
    /// word found.
    pub fn detect_words(&self, input: &OcrInput) -> Result<Vec<RotatedRect>, Box<dyn Error>> {
        let Some(detection_model) = self.detection_model.as_ref() else {
            return Err("Detection model not loaded".into());
        };
        detect_words(input.image.view(), detection_model, self.debug)
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
        _input: &OcrInput,
        words: &[RotatedRect],
    ) -> Vec<Vec<RotatedRect>> {
        find_text_lines(words)
    }

    /// Recognize lines of text in an image.
    ///
    /// `lines` is an ordered list of the text line boxes in an image,
    /// produced by [OcrEngine::find_text_lines].
    ///
    /// The output is a list of [TextLine]s corresponding to the input image
    /// regions. Entries can be `None` if no text was found in a given line.
    pub fn recognize_text(
        &self,
        input: &OcrInput,
        lines: &[Vec<RotatedRect>],
    ) -> Result<Vec<Option<TextLine>>, Box<dyn Error>> {
        let Some(recognition_model) = self.recognition_model.as_ref() else {
            return Err("Recognition model not loaded".into());
        };
        recognize_text_lines(
            input.image.view(),
            lines,
            recognition_model,
            RecognitionOpt {
                debug: self.debug,
                decode_method: self.decode_method,
            },
        )
    }

    /// Convenience API that extracts all text from an image as a single string.
    pub fn get_text(&self, input: &OcrInput) -> Result<String, Box<dyn Error>> {
        let word_rects = self.detect_words(input)?;
        let line_rects = self.find_text_lines(input, &word_rects);
        let text = self
            .recognize_text(input, &line_rects)?
            .into_iter()
            .filter_map(|line| line.map(|l| l.to_string()))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use wasnn::ops::{MaxPool, Transpose};
    use wasnn::Model;
    use wasnn::{Dimension, ModelBuilder, OpType};
    use wasnn_imageproc::{fill_rect, BoundingRect, Rect, RectF, RotatedRect};
    use wasnn_tensor::{Layout, NdTensor, Tensor};

    use super::{OcrEngine, OcrEngineParams};

    /// Generate a dummy CHW input image for OCR processing.
    ///
    /// The result is an RGB image which is black except for one line containing
    /// `n_words` white-filled rects.
    fn gen_test_image(n_words: usize) -> NdTensor<f32, 3> {
        let mut image = NdTensor::zeros([3, 100, 200]);

        for word_idx in 0..n_words {
            for chan_idx in 0..3 {
                fill_rect(
                    image.slice_mut([chan_idx]),
                    Rect::from_tlhw(30, (word_idx * 70) as i32, 20, 50),
                    1.,
                );
            }
        }

        image
    }

    /// Create a fake text detection model.
    ///
    /// Takes a CHW input tensor with values in `[-0.5, 0.5]` and adds a +0.5
    /// bias to produce an output "probability map".
    fn fake_detection_model() -> Model {
        let mut mb = ModelBuilder::new();
        let input_id = mb.add_value(
            "input",
            Some(&[
                Dimension::Symbolic("batch".to_string()),
                Dimension::Fixed(1),
                // The real model uses larger inputs (800x600). The fake uses
                // smaller inputs to make tests run faster.
                Dimension::Fixed(200),
                Dimension::Fixed(100),
            ]),
        );
        mb.add_input(input_id);

        let output_id = mb.add_value("output", None);
        mb.add_output(output_id);

        let bias = Tensor::from_scalar(0.5);
        let bias_id = mb.add_float_constant(&bias);
        mb.add_operator(
            "add",
            OpType::Add,
            &[Some(input_id), Some(bias_id)],
            &[output_id],
        );

        let model_data = mb.finish();
        Model::load(&model_data).unwrap()
    }

    /// Create a fake text recognition model.
    ///
    /// This takes an NCHW input with C=1, H=64 and returns an output with
    /// shape `[W / 4, N, C]`. In the real model the last dimension is the
    /// log-probability of each class label. In this fake we just re-interpret
    /// each column of the input as a one-hot vector of probabilities.
    fn fake_recognition_model() -> Model {
        let mut mb = ModelBuilder::new();
        let input_id = mb.add_value(
            "input",
            Some(&[
                Dimension::Symbolic("batch".to_string()),
                Dimension::Fixed(1),
                Dimension::Fixed(64),
                Dimension::Symbolic("seq".to_string()),
            ]),
        );
        mb.add_input(input_id);

        // MaxPool to scale width by 1/4: NCHW => NCHW/4
        let pool_out = mb.add_value("max_pool_out", None);
        mb.add_operator(
            "max_pool",
            OpType::MaxPool(MaxPool {
                kernel_size: [1, 4],
                padding: [0, 0, 0, 0].into(),
                strides: [1, 4],
            }),
            &[Some(input_id)],
            &[pool_out],
        );

        // Squeeze to remove the channel dim: NCHW/4 => NHW/4
        let squeeze_axes = Tensor::from_vec(vec![1]);
        let squeeze_axes_id = mb.add_int_constant(&squeeze_axes);
        let squeeze_out = mb.add_value("squeeze_out", None);
        mb.add_operator(
            "squeeze",
            OpType::Squeeze,
            &[Some(pool_out), Some(squeeze_axes_id)],
            &[squeeze_out],
        );

        // Transpose: NHW/4 => W/4NH
        let transpose_out = mb.add_value("transpose_out", None);
        mb.add_operator(
            "transpose",
            OpType::Transpose(Transpose {
                perm: Some(vec![2, 0, 1]),
            }),
            &[Some(squeeze_out)],
            &[transpose_out],
        );

        mb.add_output(transpose_out);

        let model_data = mb.finish();
        Model::load(&model_data).unwrap()
    }

    /// Return expected word locations for an image generated by
    /// `gen_test_image(3)`.
    ///
    /// The output boxes are slightly larger than in input image. This is
    /// because the real detection model is trained to predict boxes that are
    /// slightly smaller than the ground truth, in order to create a gap between
    /// adjacent boxes. The connected components in model outputs are then
    /// expanded in post-processing to recover the correct boxes.
    fn expected_word_boxes() -> Vec<RectF> {
        let [top, height] = [27, 25];
        [
            Rect::from_tlhw(top, -3, height, 56).to_f32(),
            Rect::from_tlhw(top, 66, height, 57).to_f32(),
            Rect::from_tlhw(top, 136, height, 57).to_f32(),
        ]
        .into()
    }

    #[test]
    fn test_ocr_engine_prepare_input() -> Result<(), Box<dyn Error>> {
        let image = gen_test_image(3 /* n_words */);
        let engine = OcrEngine::new(OcrEngineParams {
            detection_model: None,
            recognition_model: None,
            ..Default::default()
        })?;
        let input = engine.prepare_input(image.view())?;

        let [chans, height, width] = input.image.shape();
        assert_eq!(chans, 1);
        assert_eq!(width, image.size(2));
        assert_eq!(height, image.size(1));

        Ok(())
    }

    #[test]
    fn test_ocr_engine_detect_words() -> Result<(), Box<dyn Error>> {
        let n_words = 3;
        let image = gen_test_image(n_words);
        let engine = OcrEngine::new(OcrEngineParams {
            detection_model: Some(fake_detection_model()),
            recognition_model: None,
            ..Default::default()
        })?;
        let input = engine.prepare_input(image.view())?;
        let words = engine.detect_words(&input)?;

        assert_eq!(words.len(), n_words);

        let mut boxes: Vec<RectF> = words
            .into_iter()
            .map(|rotated_rect| rotated_rect.bounding_rect())
            .collect();
        boxes.sort_by_key(|b| [b.top() as i32, b.left() as i32]);

        assert_eq!(boxes, expected_word_boxes());

        Ok(())
    }

    #[test]
    fn test_ocr_engine_recognize_lines() -> Result<(), Box<dyn Error>> {
        let mut image = NdTensor::zeros([1, 64, 32]);

        // Fill a single row of the input image.
        //
        // The dummy recognition model treats each column of the input as a
        // one-hot vector of character class probabilities. Pre-processing of
        // the input will shift values from [0, 1] to [-0.5, 0.5]. CTC decoding
        // of the output will ignore class 0 (as it represents a CTC blank)
        // and repeated characters.
        //
        // Filling a single input row with "1"s will produce a single char
        // output where the char's index in the alphabet is the row index - 1.
        // ie. Filling the first row produces " ", the second row "0" and so on,
        // using the default alphabet.
        image
            .slice_mut::<2, 3, _>((.., 2, ..))
            .iter_mut()
            .for_each(|x| *x = 1.);

        let engine = OcrEngine::new(OcrEngineParams {
            detection_model: None,
            recognition_model: Some(fake_recognition_model()),
            ..Default::default()
        })?;
        let input = engine.prepare_input(image.view())?;

        // Create a dummy input line with a single word which fills the image.
        let mut line_regions: Vec<Vec<RotatedRect>> = Vec::new();
        line_regions.push(
            [Rect::from_tlhw(0, 0, image.shape()[1] as i32, image.shape()[2] as i32).to_f32()]
                .map(RotatedRect::from_rect)
                .into(),
        );

        let lines = engine.recognize_text(&input, &line_regions)?;
        assert_eq!(lines.len(), line_regions.len());

        assert!(lines.get(0).is_some());
        let line = lines[0].as_ref().unwrap();
        assert_eq!(line.to_string(), "0");

        Ok(())
    }
}
