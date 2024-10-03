use core::f32;
use std::collections::HashMap;

use anyhow::anyhow;
use rayon::prelude::*;
use rten::ctc::{CtcDecoder, CtcHypothesis};
use rten::{thread_pool, Dimension, FloatOperators, Model, NodeId};
use rten_imageproc::{
    bounding_rect, BoundingRect, Line, Point, PointF, Polygon, Rect, RotatedRect,
};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, NdTensorViewMut, Tensor};

use crate::errors::ModelRunError;
use crate::geom_util::{downwards_line, leftmost_edge, rightmost_edge};
use crate::preprocess::BLACK_VALUE;
use crate::text_items::{TextChar, TextLine};

/// Return a polygon which contains all the rects in `words`.
///
/// `words` is assumed to be a series of disjoint rectangles ordered from left
/// to right. The returned points are arranged in clockwise order starting from
/// the top-left point.
///
/// There are several ways to compute a polygon for a line. The simplest is
/// to use [min_area_rect] on the union of the line's points. However the result
/// will not tightly fit curved lines. This function returns a polygon which
/// closely follows the edges of individual words.
fn line_polygon(words: &[RotatedRect]) -> Vec<Point> {
    let mut polygon = Vec::new();

    let floor_point = |p: PointF| Point::from_yx(p.y as i32, p.x as i32);

    // Add points from top edges, in left-to-right order.
    for word_rect in words.iter() {
        let (left, right) = (
            downwards_line(leftmost_edge(word_rect)),
            downwards_line(rightmost_edge(word_rect)),
        );
        polygon.push(floor_point(left.start));
        polygon.push(floor_point(right.start));
    }

    // Add points from bottom edges, in right-to-left order.
    for word_rect in words.iter().rev() {
        let (left, right) = (
            downwards_line(leftmost_edge(word_rect)),
            downwards_line(rightmost_edge(word_rect)),
        );
        polygon.push(floor_point(right.end));
        polygon.push(floor_point(left.end));
    }

    polygon
}

/// Compute width to resize a text line image to, for a given height.
fn resized_line_width(orig_width: i32, orig_height: i32, height: i32) -> u32 {
    let min_width = 10.;

    // A larger maximum width avoids horizontally squashing long input lines,
    // affecting accuracy. However it also increases the processing time.
    //
    // The current value was chosen to be large enough to produce good results
    // on screenshots taken from the longest lines in English Wikipedia articles
    // (image size approx 1860x30, 150 characters).
    //
    // The widest image seen during training may be constrained to a shorter
    // value than this, but we rely on the model's ability to generalize to
    // longer sequences.
    let max_width = 2400.;

    let aspect_ratio = orig_width as f32 / orig_height as f32;
    (height as f32 * aspect_ratio).clamp(min_width, max_width) as u32
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

fn prepare_text_line(
    image: NdTensorView<f32, 3>,
    page_rect: Rect,
    line_region: &Polygon,
    resized_width: u32,
    output_height: usize,
) -> NdTensor<f32, 2> {
    // Page rect adjusted to only contain coordinates that are valid for
    // indexing into the input image.
    let page_index_rect = page_rect.adjust_tlbr(0, 0, -1, -1);

    let grey_chan = image.slice([0]);

    let line_rect = line_region.bounding_rect();
    let mut line_img = NdTensor::full(
        [line_rect.height() as usize, line_rect.width() as usize],
        BLACK_VALUE,
    );

    for in_p in line_region.fill_iter() {
        let out_p = Point::from_yx(in_p.y - line_rect.top(), in_p.x - line_rect.left());
        if !page_index_rect.contains_point(in_p) || !page_index_rect.contains_point(out_p) {
            continue;
        }
        line_img[[out_p.y as usize, out_p.x as usize]] =
            grey_chan[[in_p.y as usize, in_p.x as usize]];
    }

    let resized_line_img = line_img
        .reshaped([1, 1, line_img.size(0), line_img.size(1)])
        .resize_image([output_height, resized_width as usize])
        .unwrap();

    let out_shape = [resized_line_img.size(2), resized_line_img.size(3)];
    resized_line_img.into_shape(out_shape)
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
    let mut output = NdTensor::full([lines.len(), 1, output_height, output_width], BLACK_VALUE);

    for (group_line_index, line) in lines.iter().enumerate() {
        let resized_line_img = prepare_text_line(
            image.view(),
            page_rect,
            &line.region,
            line.resized_width,
            output_height,
        );
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
                .map_or(e.start, |y| Point::from_yx(y.round() as i32, min_x));

            let trunc_edge_end = e
                .to_f32()
                .y_for_x(max_x as f32)
                .map_or(e.end, |y| Point::from_yx(y.round() as i32, max_x));

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
pub struct RecognitionOpt<'a> {
    pub debug: bool,

    /// Method used to decode character sequence outputs to character values.
    pub decode_method: DecodeMethod,

    pub alphabet: &'a str,

    pub excluded_char_labels: Option<&'a [usize]>,
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
fn text_lines_from_recognition_results(
    results: &[LineRecResult],
    alphabet: &str,
) -> Vec<Option<TextLine>> {
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

                    let char = alphabet
                        .chars()
                        // Index `0` is reserved for blank character and `i + 1` is used as training
                        // label for character at index `i` of `alphabet` string.  Here we're
                        // subtracting 1 to get the actual index from the output label
                        //
                        // See https://github.com/robertknight/ocrs-models/blob/3d98fc655d6fd4acddc06e7f5d60a55b55748a48/ocrs_models/datasets/util.py#L113
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

/// Extracts character sequences and coordinates from text lines detected in
/// an image.
pub struct TextRecognizer {
    model: Model,
    input_id: NodeId,
    input_shape: Vec<Dimension>,
    output_id: NodeId,
}

impl TextRecognizer {
    /// Initialize a text recognizer from a trained RTen model. Fails if the
    /// model does not have the expected inputs or outputs.
    pub fn from_model(model: Model) -> anyhow::Result<TextRecognizer> {
        let input_id = model
            .input_ids()
            .first()
            .copied()
            .ok_or(anyhow!("recognition model has no inputs"))?;
        let input_shape = model
            .node_info(input_id)
            .and_then(|info| info.shape())
            .ok_or(anyhow!("recognition model does not specify input shape"))?;
        let output_id = model
            .output_ids()
            .first()
            .copied()
            .ok_or(anyhow!("recognition model has no outputs"))?;
        Ok(TextRecognizer {
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
    fn run(&self, input: NdTensor<f32, 4>) -> Result<NdTensor<f32, 3>, ModelRunError> {
        let input: Tensor<f32> = input.into();
        let [output] = self
            .model
            .run_n(
                vec![(self.input_id, (&input).into())],
                [self.output_id],
                None,
            )
            .map_err(|err| ModelRunError::RunFailed(err.into()))?;

        let output_ndim = output.ndim();
        let mut rec_sequence: NdTensor<f32, 3> = output.try_into().map_err(|_| {
            ModelRunError::WrongOutput(format!(
                "expected recognition output to have 3 dims but it has {}",
                output_ndim
            ))
        })?;

        // Transpose from [seq, batch, class] => [batch, seq, class]
        rec_sequence.permute([1, 0, 2]);

        Ok(rec_sequence)
    }

    /// Prepare a text line for input into the recognition model.
    ///
    /// This method exists for model debugging purposes to expose the
    /// preprocessing that [TextRecognizer::recognize_text_lines] does.
    pub fn prepare_input(
        &self,
        image: NdTensorView<f32, 3>,
        line: &[RotatedRect],
    ) -> NdTensor<f32, 2> {
        // These lines should match corresponding code in
        // `recognize_text_lines`.
        let [_, img_height, img_width] = image.shape();
        let page_rect = Rect::from_hw(img_height as i32, img_width as i32);

        let line_rect = bounding_rect(line.iter())
            .expect("line has no words")
            .integral_bounding_rect();

        let line_poly = Polygon::new(line_polygon(line));
        let rec_img_height = self.input_height();
        let resized_width =
            resized_line_width(line_rect.width(), line_rect.height(), rec_img_height as i32);

        prepare_text_line(
            image,
            page_rect,
            &line_poly,
            resized_width,
            rec_img_height as usize,
        )
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
    pub fn recognize_text_lines(
        &self,
        image: NdTensorView<f32, 3>,
        lines: &[Vec<RotatedRect>],
        opts: RecognitionOpt,
    ) -> anyhow::Result<Vec<Option<TextLine>>> {
        let RecognitionOpt {
            debug,
            decode_method,
            alphabet,
            excluded_char_labels,
        } = opts;

        let [_, img_height, img_width] = image.shape();
        let page_rect = Rect::from_hw(img_height as i32, img_width as i32);

        // Group lines into batches which will have similar widths after resizing
        // to a fixed height.
        //
        // It is more efficient to run recognition on multiple lines at once, but
        // all line images in a batch must be padded to an equal length. Some
        // computation is wasted on shorter lines in the batch. Choosing batches
        // such that all line images have a similar width reduces this wastage.
        // There is a trade-off between maximizing the batch size and minimizing
        // the variance in width of images in the batch.
        let rec_img_height = self.input_height();
        let mut line_groups: HashMap<i32, Vec<TextRecLine>> = HashMap::new();
        for (line_index, word_rects) in lines.iter().enumerate() {
            let line_rect = bounding_rect(word_rects.iter())
                .expect("line has no words")
                .integral_bounding_rect();
            let resized_width =
                resized_line_width(line_rect.width(), line_rect.height(), rec_img_height as i32);
            let group_width = resized_width.next_multiple_of(50);
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

        let alphabet_len = alphabet.chars().count();

        // Run text recognition on batches of lines.
        let batch_rec_results: Result<Vec<Vec<LineRecResult>>, ModelRunError> =
            thread_pool().run(|| {
                line_groups
                    .into_par_iter()
                    .map(|(group_width, lines)| {
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

                        let mut rec_output = self.run(rec_input)?;

                        if alphabet_len + 1 != rec_output.size(2) {
                            return Err(ModelRunError::WrongOutput(format!(
                                "output column count ({}) does not match alphabet size ({})",
                                rec_output.size(2),
                                alphabet_len + 1
                            )));
                        }

                        let ctc_input_len = rec_output.shape()[1];

                        // Apply CTC decoding to get the label sequence for each line.
                        let line_rec_results = lines
                            .into_iter()
                            .enumerate()
                            .map(|(group_line_index, line)| {
                                let decoder = CtcDecoder::new();

                                let mut input_seq_slice = rec_output.slice_mut([group_line_index]);
                                let input_seq = Self::filter_excluded_char_labels(
                                    excluded_char_labels,
                                    &mut input_seq_slice,
                                );

                                let ctc_output = match decode_method {
                                    DecodeMethod::Greedy => decoder.decode_greedy(input_seq),
                                    DecodeMethod::BeamSearch { width } => {
                                        decoder.decode_beam(input_seq, width)
                                    }
                                };
                                LineRecResult {
                                    line,
                                    rec_input_len: group_width as usize,
                                    ctc_input_len,
                                    ctc_output,
                                }
                            })
                            .collect::<Vec<_>>();

                        Ok(line_rec_results)
                    })
                    .collect()
            });

        let mut line_rec_results: Vec<LineRecResult> =
            batch_rec_results?.into_iter().flatten().collect();

        // The recognition outputs are in a different order than the inputs due to
        // batching and parallel processing. Re-sort them into input order.
        line_rec_results.sort_by_key(|result| result.line.index);

        let text_lines = text_lines_from_recognition_results(&line_rec_results, alphabet);

        Ok(text_lines)
    }

    /// Post-process recognition model outputs to filter excluded characters.
    ///
    /// `input_seq_slice` is a (seq, char_prob) matrix of log probabilities for
    /// characters. `excluded_char_labels` specifies indices of characters that
    /// should be excluded, by setting the log probability to -Inf.
    fn filter_excluded_char_labels<'a>(
        excluded_char_labels: Option<&[usize]>,
        input_seq_slice: &'a mut NdTensorViewMut<'_, f32, 2>,
    ) -> NdTensorView<'a, f32, 2> {
        if let Some(excluded_char_labels) = excluded_char_labels {
            for row in 0..input_seq_slice.size(0) {
                for &excluded_char_label in excluded_char_labels.iter() {
                    // Setting the output value of excluded char to -Inf causes the
                    // `decode_method` to favour chars other than the excluded char.
                    (*input_seq_slice)[[row, excluded_char_label]] = f32::NEG_INFINITY;
                }
            }
        }
        input_seq_slice.view()
    }
}

#[cfg(test)]
mod tests {
    use rten_imageproc::{BoundingRect, Point, Polygon, RotatedRect, Vec2};

    use super::line_polygon;

    #[test]
    fn test_line_polygon() {
        let words: Vec<RotatedRect> = (0..5)
            .map(|i| {
                let center = Point::from_yx(10., i as f32 * 20.);
                let width = 10.;
                let height = 5.;

                // Vary the orientation of words. The output of `line_polygon`
                // should be invariant to different orientations of a RotatedRect
                // that cover the same pixels.
                let up = Vec2::from_yx(if i % 2 == 0 { -1. } else { 1. }, 0.);
                RotatedRect::new(center, up, width, height)
            })
            .collect();
        let poly = Polygon::new(line_polygon(&words));

        assert!(poly.is_simple());
        for word in words {
            let center = word.bounding_rect().center();
            assert!(poly.contains_pixel(Point::from_yx(
                center.y.round() as i32,
                center.x.round() as i32
            )));
        }
    }
}
