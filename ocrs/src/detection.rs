use anyhow::anyhow;
use rten::{Dimension, FloatOperators, Model, Operators, RunOptions};
use rten_imageproc::{find_contours, min_area_rect, simplify_polygon, RetrievalMode, RotatedRect};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor};

use crate::preprocess::BLACK_VALUE;
use crate::tensor_util::IntoCow;

/// Parameters that control post-processing of text detection model outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct TextDetectorParams {
    /// Threshold for minimum area of returned rectangles.
    ///
    /// This can be used to filter out rects created by small false positives in
    /// the mask, at the risk of filtering out true positives. The more accurate
    /// the model producing the mask is, the smaller this value can be.
    pub min_area: f32,

    /// Threshold for per-pixel scores in output segmentation mask for
    /// classifying a pixel as text.
    pub text_threshold: f32,
}

impl Default for TextDetectorParams {
    fn default() -> TextDetectorParams {
        // Empirically chosen parameters for the initial model release.
        TextDetectorParams {
            // This area is quite large and can prevent detection of small /
            // single letter words.
            min_area: 100.,

            // Ideally the threshold would be 0.5 as a neutral value.
            text_threshold: 0.2,
        }
    }
}

/// Find the minimum-area oriented rectangles containing each connected
/// component in the binary mask `mask`.
fn find_connected_component_rects(
    mask: NdTensorView<i32, 2>,
    expand_dist: f32,
    min_area: f32,
) -> Vec<RotatedRect> {
    find_contours(mask, RetrievalMode::External)
        .iter()
        .filter_map(|poly| {
            let float_points: Vec<_> = poly.iter().map(|p| p.to_f32()).collect();
            let simplified = simplify_polygon(&float_points, 2. /* epsilon */);

            min_area_rect(&simplified).map(|mut rect| {
                rect.resize(
                    rect.width() + 2. * expand_dist,
                    rect.height() + 2. * expand_dist,
                );
                rect
            })
        })
        .filter(|r| r.area() >= min_area)
        .collect()
}

/// Text detector which finds the oriented bounding boxes of words in an input
/// image.
pub struct TextDetector {
    model: Model,
    params: TextDetectorParams,
    input_shape: Vec<Dimension>,
}

impl TextDetector {
    /// Initializate a DetectionModel from a trained RTen model.
    ///
    /// This will fail if the model doesn't have the expected inputs or outputs.
    pub fn from_model(model: Model, params: TextDetectorParams) -> anyhow::Result<TextDetector> {
        let input_id = model
            .input_ids()
            .first()
            .copied()
            .ok_or(anyhow!("model has no inputs"))?;
        let input_shape = model
            .node_info(input_id)
            .and_then(|info| info.shape())
            .ok_or(anyhow!("model does not specify expected input shape"))?;

        Ok(TextDetector {
            model,
            params,
            input_shape,
        })
    }

    /// Return the confidence threshold used to determine whether a pixel is
    /// text or not.
    pub fn threshold(&self) -> f32 {
        self.params.text_threshold
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
    pub fn detect_words(
        &self,
        image: NdTensorView<f32, 3>,
        debug: bool,
    ) -> anyhow::Result<Vec<RotatedRect>> {
        let text_mask = self.detect_text_pixels(image, debug)?;
        let binary_mask = text_mask.map(|prob| {
            if *prob > self.params.text_threshold {
                1i32
            } else {
                0
            }
        });

        // Distance to expand bounding boxes by. This is useful when the model is
        // trained to assign a positive label to pixels in a smaller area than the
        // ground truth, which may be done to create separation between adjacent
        // objects.
        let expand_dist = 3.;

        let word_rects =
            find_connected_component_rects(binary_mask.view(), expand_dist, self.params.min_area);

        Ok(word_rects)
    }

    /// Detect text pixels in an image.
    ///
    /// Takes a greyscale (CHW) input image and returns a probability map
    /// indicating whether each pixel in the input is text.
    ///
    /// See [detect_words](TextDetector::detect_words) for more details of
    /// expected input.
    pub fn detect_text_pixels(
        &self,
        image: NdTensorView<f32, 3>,
        debug: bool,
    ) -> anyhow::Result<NdTensor<f32, 2>> {
        let [img_chans, img_height, img_width] = image.shape();

        // Add batch dim
        let image = image.reshaped([1, img_chans, img_height, img_width]);

        let [_, _, Dimension::Fixed(in_height), Dimension::Fixed(in_width)] = self.input_shape[..]
        else {
            return Err(anyhow!("failed to get model dims"));
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
        let image = (pad_bottom > 0 || pad_right > 0)
            .then(|| {
                let pads = &[0, 0, 0, 0, 0, 0, pad_bottom, pad_right];
                image.pad(pads.into(), BLACK_VALUE)
            })
            .transpose()?
            .map(|t| t.into_cow())
            .unwrap_or(image.into_dyn().into_cow());

        // Resize images to the text detection model's input size.
        let image = (image.size(2) != in_height || image.size(3) != in_width)
            .then(|| image.resize_image([in_height, in_width]))
            .transpose()?
            .map(|t| t.into_cow())
            .unwrap_or(image);

        // Run text detection model to compute a probability mask indicating whether
        // each pixel is part of a text word or not.
        let text_mask: Tensor<f32> = self
            .model
            .run_one(
                image.view().into(),
                if debug {
                    Some(RunOptions {
                        timing: true,
                        verbose: false,
                        ..Default::default()
                    })
                } else {
                    None
                },
            )?
            .try_into()?;

        // Resize probability mask to original input size and apply threshold to get a
        // binary text/not-text mask.
        let text_mask = text_mask
            .slice::<4, _>((
                ..,
                ..,
                ..(in_height - pad_bottom as usize),
                ..(in_width - pad_right as usize),
            ))
            .resize_image([img_height, img_width])?;

        // Remove batch, channel dims.
        let text_mask = text_mask.into_shape([img_height, img_width]);

        Ok(text_mask)
    }
}

#[cfg(test)]
mod tests {
    use rten_imageproc::{fill_rect, Point};
    use rten_tensor::prelude::*;
    use rten_tensor::NdTensor;

    use super::find_connected_component_rects;
    use crate::test_util::gen_rect_grid;

    #[test]
    fn test_find_connected_component_rects() {
        let mut mask = NdTensor::zeros([400, 400]);
        let (grid_h, grid_w) = (5, 5);
        let (rect_h, rect_w) = (10, 50);
        let rects = gen_rect_grid(
            Point::from_yx(10, 10),
            (grid_h, grid_w), /* grid_shape */
            (rect_h, rect_w), /* rect_size */
            (10, 5),          /* gap_size */
        );
        for r in rects.iter() {
            // Expand `r` because `fill_rect` does not set points along the
            // right/bottom boundary.
            let expanded = r.adjust_tlbr(0, 0, 1, 1);
            fill_rect(mask.view_mut(), expanded, 1);
        }

        let min_area = 100.;
        let components = find_connected_component_rects(mask.view(), 0., min_area);
        assert_eq!(components.len() as i32, grid_h * grid_w);

        for c in components.iter() {
            let mut shape = [c.height().round() as i32, c.width().round() as i32];
            shape.sort();

            // We sort the dimensions before comparison here to be invariant to
            // different rotations of the connected component that cover the
            // same pixels.
            let mut expected_shape = [rect_h, rect_w];
            expected_shape.sort();

            assert_eq!(shape, expected_shape);
        }
    }
}
