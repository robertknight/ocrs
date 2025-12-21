use std::fmt::Debug;

use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};
use thiserror::Error;

/// View of an image's pixels, in either (height, width, channels) or (channels,
/// height, width) order.
pub enum ImagePixels<'a> {
    /// Pixel values in the range [0, 1]
    Floats(NdTensorView<'a, f32, 3>),
    /// Pixel values in the range [0, 255]
    Bytes(NdTensorView<'a, u8, 3>),
}

impl<'a> From<NdTensorView<'a, f32, 3>> for ImagePixels<'a> {
    fn from(value: NdTensorView<'a, f32, 3>) -> Self {
        ImagePixels::Floats(value)
    }
}

impl<'a> From<NdTensorView<'a, u8, 3>> for ImagePixels<'a> {
    fn from(value: NdTensorView<'a, u8, 3>) -> Self {
        ImagePixels::Bytes(value)
    }
}

impl ImagePixels<'_> {
    fn shape(&self) -> [usize; 3] {
        match self {
            ImagePixels::Floats(f) => f.shape(),
            ImagePixels::Bytes(b) => b.shape(),
        }
    }
}

/// Errors that can occur when creating an [ImageSource].
#[derive(Error, Clone, Debug, PartialEq)]
pub enum ImageSourceError {
    /// The image channel count is not 1 (greyscale), 3 (RGB) or 4 (RGBA).
    #[error("channel count is not 1, 3 or 4")]
    UnsupportedChannelCount,
    /// The image data length is not a multiple of the channel size.
    #[error("data length is not a multiple of `width * height`")]
    InvalidDataLength,
}

/// Specifies the order in which pixels are laid out in an image tensor.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DimOrder {
    /// Channels last order. This is the order used by the
    /// [image](https://github.com/image-rs/image) crate and HTML Canvas APIs.
    Hwc,
    /// Channels first order. This is the order used by many machine-learning
    /// libraries for image tensors.
    Chw,
}

/// View of an image, for use with
/// [OcrEngine::prepare_input](crate::OcrEngine::prepare_input).
pub struct ImageSource<'a> {
    data: ImagePixels<'a>,
    order: DimOrder,
}

impl<'a> ImageSource<'a> {
    /// Create an image source from a buffer of pixels in HWC order.
    ///
    /// An image loaded using the `image` crate can be converted to an
    /// [ImageSource] using:
    ///
    /// ```no_run
    /// use ocrs::ImageSource;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let image = image::open("image.jpg")?.into_rgb8();
    /// let img_source = ImageSource::from_bytes(image.as_raw(), image.dimensions())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_bytes(
        bytes: &'a [u8],
        dimensions: (u32, u32),
    ) -> Result<ImageSource<'a>, ImageSourceError> {
        let (width, height) = dimensions;
        let channel_len = (width * height) as usize;

        if channel_len == 0 {
            return Err(ImageSourceError::UnsupportedChannelCount);
        }

        if !bytes.len().is_multiple_of(channel_len) {
            return Err(ImageSourceError::InvalidDataLength);
        }

        let channels = bytes.len() / channel_len;
        Self::from_tensor(
            NdTensorView::from_data([height as usize, width as usize, channels], bytes),
            DimOrder::Hwc,
        )
    }

    /// Create an image source from a tensor of bytes (`u8`) or floats (`f32`),
    /// in either channels-first (CHW) or channels-last (HWC) order.
    pub fn from_tensor<T>(
        data: NdTensorView<'a, T, 3>,
        order: DimOrder,
    ) -> Result<ImageSource<'a>, ImageSourceError>
    where
        NdTensorView<'a, T, 3>: Into<ImagePixels<'a>>,
    {
        let channels = match order {
            DimOrder::Hwc => data.size(2),
            DimOrder::Chw => data.size(0),
        };
        match channels {
            1 | 3 | 4 => Ok(ImageSource {
                data: data.into(),
                order,
            }),
            _ => Err(ImageSourceError::UnsupportedChannelCount),
        }
    }
}

/// The value used to represent fully black pixels in OCR input images
/// prepared by [prepare_image].
pub const BLACK_VALUE: f32 = -0.5;

/// Specifies the number and order of color channels in an image.
enum Channels {
    Grey,
    Rgb,
    Rgba,
}

/// Prepare an image for use with text detection and recognition models.
///
/// This involves:
///
/// - Converting the pixels to floats
/// - Converting the color format to greyscale
/// - Adding a bias ([BLACK_VALUE]) to the greyscale value
///
/// The greyscale conversion is intended to approximately match torchvision's
/// RGB => greyscale conversion when using `torchvision.io.read_image(path,
/// ImageReadMode.GRAY)`, which is used when training models with greyscale
/// inputs. torchvision internally uses libpng's `png_set_rgb_to_gray`.
pub fn prepare_image(img: ImageSource) -> NdTensor<f32, 3> {
    match img.order {
        DimOrder::Hwc => prepare_image_impl::<true>(img.data),
        DimOrder::Chw => prepare_image_impl::<false>(img.data),
    }
}

fn prepare_image_impl<const CHANS_LAST: bool>(pixels: ImagePixels) -> NdTensor<f32, 3> {
    let n_chans = if CHANS_LAST {
        pixels.shape()[2]
    } else {
        pixels.shape()[0]
    };
    let src_chans = match n_chans {
        1 => Channels::Grey,
        3 => Channels::Rgb,
        4 => Channels::Rgba,
        _ => panic!("expected greyscale, RGB or RGBA input image"),
    };

    // ITU BT.601 weights for RGB => luminance conversion. These match what
    // torchvision uses. See also https://stackoverflow.com/a/596241/434243.
    const ITU_WEIGHTS: [f32; 3] = [0.299, 0.587, 0.114];

    match pixels {
        ImagePixels::Floats(floats) => match src_chans {
            Channels::Grey => convert_pixels::<_, _, CHANS_LAST>(floats.view(), [1.]),
            Channels::Rgb | Channels::Rgba => {
                convert_pixels::<_, _, CHANS_LAST>(floats.view(), ITU_WEIGHTS)
            }
        },
        ImagePixels::Bytes(bytes) => match src_chans {
            Channels::Grey => convert_pixels::<_, _, CHANS_LAST>(bytes.view(), [1. / 255.]),
            Channels::Rgb | Channels::Rgba => {
                // Combine the byte -> float scaling and color components into
                // a single weight.
                let weights = ITU_WEIGHTS.map(|w| w / 255.0);
                convert_pixels::<_, _, CHANS_LAST>(bytes.view(), weights)
            }
        },
    }
}

/// Convert pixels in an image to floats and scale by the given channel weights.
///
/// Returns a (1, H, W) tensor.
fn convert_pixels<T: AsF32, const N: usize, const CHANS_LAST: bool>(
    src: NdTensorView<T, 3>,
    chan_weights: [f32; N],
) -> NdTensor<f32, 3> {
    let height = if CHANS_LAST { src.size(0) } else { src.size(1) };
    let width = if CHANS_LAST { src.size(1) } else { src.size(2) };

    let mut grey_img = NdTensor::uninit([height, width]);
    for y in 0..height {
        for x in 0..width {
            let mut pixel = BLACK_VALUE;
            for c in 0..chan_weights.len() {
                let src_elem = if CHANS_LAST {
                    src[[y, x, c]].as_f32()
                } else {
                    src[[c, y, x]].as_f32()
                };
                pixel += src_elem * chan_weights[c]
            }
            grey_img[[y, x]].write(pixel);
        }
    }

    // Safety: We initialized all the pixels.
    unsafe { grey_img.assume_init().into_shape([1, height, width]) }
}

/// Convert a primitive to a float using the `as` operator.
trait AsF32: Copy {
    fn as_f32(self) -> f32;
}

impl AsF32 for f32 {
    fn as_f32(self) -> f32 {
        self
    }
}

impl AsF32 for u8 {
    fn as_f32(self) -> f32 {
        self as f32
    }
}

#[cfg(test)]
mod tests {
    use rten_tensor::prelude::*;
    use rten_tensor::NdTensor;

    use super::{DimOrder, ImageSource, ImageSourceError};

    #[test]
    fn test_image_source_from_bytes() {
        struct Case {
            len: usize,
            width: u32,
            height: u32,
            error: Option<ImageSourceError>,
        }

        let cases = [
            Case {
                len: 100,
                width: 10,
                height: 10,
                error: None,
            },
            Case {
                len: 50,
                width: 10,
                height: 10,
                error: Some(ImageSourceError::InvalidDataLength),
            },
            Case {
                len: 8 * 8 * 2,
                width: 8,
                height: 8,
                error: Some(ImageSourceError::UnsupportedChannelCount),
            },
            Case {
                len: 0,
                width: 0,
                height: 10,
                error: Some(ImageSourceError::UnsupportedChannelCount),
            },
        ];

        for Case {
            len,
            width,
            height,
            error,
        } in cases
        {
            let data: Vec<u8> = (0u8..len as u8).collect();
            let source = ImageSource::from_bytes(&data, (width, height));
            assert_eq!(source.as_ref().err(), error.as_ref());
        }
    }

    #[test]
    fn test_image_source_from_data() {
        struct Case {
            shape: [usize; 3],
            error: Option<ImageSourceError>,
            order: DimOrder,
        }

        let cases = [
            Case {
                shape: [1, 5, 5],
                error: None,
                order: DimOrder::Chw,
            },
            Case {
                shape: [1, 5, 5],
                error: Some(ImageSourceError::UnsupportedChannelCount),
                order: DimOrder::Hwc,
            },
            Case {
                shape: [0, 5, 5],
                error: Some(ImageSourceError::UnsupportedChannelCount),
                order: DimOrder::Chw,
            },
        ];

        for Case {
            shape,
            error,
            order,
        } in cases
        {
            let len: usize = shape.iter().product();
            let tensor = NdTensor::<u8, 1>::arange(0, len as u8, None).into_shape(shape);
            let source = ImageSource::from_tensor(tensor.view(), order);
            assert_eq!(source.as_ref().err(), error.as_ref());
        }
    }
}
