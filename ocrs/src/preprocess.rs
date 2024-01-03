use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

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

/// Prepare an image for use with text detection and recognition models.
///
/// This converts an input CHW image with values in the range 0-1 to a greyscale
/// image with values in the range `BLACK_VALUE` to `BLACK_VALUE + 1`.
pub fn prepare_image(image: NdTensorView<f32, 3>) -> NdTensor<f32, 3> {
    greyscale_image(image, |pixel| pixel + BLACK_VALUE)
}
