use std::error::Error;

use rten::{Dimension, FloatOperators, Model, Operators, RunOptions};
use rten_imageproc::{find_contours, min_area_rect, simplify_polygon, RetrievalMode, RotatedRect};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor};

use crate::preprocess::BLACK_VALUE;

/// Find the minimum-area oriented rectangles containing each connected
/// component in the binary mask `mask`.
fn find_connected_component_rects(
    mask: NdTensorView<i32, 2>,
    expand_dist: f32,
) -> Vec<RotatedRect> {
    // Threshold for the minimum area of returned rectangles.
    //
    // This can be used to filter out rects created by small false positives in
    // the mask, at the risk of filtering out true positives. The more accurate
    // the model producing the mask is, the less this is needed.
    let min_area_threshold = 100.;

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
        .filter(|r| r.area() >= min_area_threshold)
        .collect()
}

/// Text detector which finds the oriented bounding boxes of words in an input
/// image.
pub struct TextDetector {
    model: Model,
    input_shape: Vec<Dimension>,
}

impl TextDetector {
    /// Initializate a DetectionModel from a trained RTen model.
    ///
    /// This will fail if the model doesn't have the expected inputs or outputs.
    pub fn from_model(model: Model) -> Result<TextDetector, Box<dyn Error>> {
        let input_id = model
            .input_ids()
            .first()
            .copied()
            .ok_or("model has no inputs")?;
        let input_shape = model
            .node_info(input_id)
            .and_then(|info| info.shape())
            .ok_or("model does not specify expected input shape")?;

        Ok(TextDetector { model, input_shape })
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
    ) -> Result<Vec<RotatedRect>, Box<dyn Error>> {
        let [img_chans, img_height, img_width] = image.shape();

        // Add batch dim
        let image = image.reshaped([1, img_chans, img_height, img_width]);

        let [_, _, Dimension::Fixed(in_height), Dimension::Fixed(in_width)] = self.input_shape[..]
        else {
            return Err("failed to get model dims".into());
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
            image.pad(pads.into(), BLACK_VALUE)?
        } else {
            image.as_dyn().to_tensor()
        };

        // Resize images to the text detection model's input size.
        let resized_grey_img = grey_img.resize_image([in_height, in_width])?;

        // Run text detection model to compute a probability mask indicating whether
        // each pixel is part of a text word or not.
        let text_mask: Tensor<f32> = self
            .model
            .run_one(
                (&resized_grey_img).into(),
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
            .slice((
                ..,
                ..,
                ..(in_height - pad_bottom as usize),
                ..(in_width - pad_right as usize),
            ))
            .resize_image([img_height, img_width])?;
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
}

#[cfg(test)]
mod tests {
    use rten_imageproc::{fill_rect, Point};
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

        let components = find_connected_component_rects(mask.view(), 0.);
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
