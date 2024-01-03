use std::error::Error;

use rten::{Dimension, FloatOperators, Model, Operators, RunOptions};
use rten_imageproc::RotatedRect;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensorView, Tensor};

use crate::page_layout::find_connected_component_rects;
use crate::preprocess::BLACK_VALUE;

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
        image.pad(pads.into(), BLACK_VALUE)?
    } else {
        image.as_dyn().to_tensor()
    };

    // Resize images to the text detection model's input size.
    let resized_grey_img = grey_img.resize_image([in_height, in_width])?;

    // Run text detection model to compute a probability mask indicating whether
    // each pixel is part of a text word or not.
    let text_mask: Tensor<f32> = model
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
