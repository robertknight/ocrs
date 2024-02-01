use std::error::Error;

use rten::Model;
use rten_imageproc::RotatedRect;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

mod detection;
mod geom_util;
mod layout_analysis;
mod log;
mod preprocess;
mod recognition;

#[cfg(test)]
mod test_util;

mod text_items;

#[cfg(target_arch = "wasm32")]
mod wasm_api;

use detection::TextDetector;
use layout_analysis::find_text_lines;
use preprocess::prepare_image;
use recognition::{RecognitionOpt, TextRecognizer};

pub use recognition::DecodeMethod;
pub use text_items::{TextChar, TextItem, TextLine, TextWord};

/// Configuration for an [OcrEngine] instance.
#[derive(Default)]
pub struct OcrEngineParams {
    /// Model used to detect text words in the image.
    pub detection_model: Option<Model>,

    /// Model used to recognize lines of text in the image.
    pub recognition_model: Option<Model>,

    /// Enable debug logging.
    pub debug: bool,

    /// Method used to decode outputs of text recognition model.
    pub decode_method: DecodeMethod,
}

/// Detects and recognizes text in images.
///
/// OcrEngine uses machine learning models to detect text, analyze layout
/// and recognize text in an image.
pub struct OcrEngine {
    detector: Option<TextDetector>,
    recognizer: Option<TextRecognizer>,
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
        let detector = params
            .detection_model
            .map(|model| TextDetector::from_model(model, Default::default()))
            .transpose()?;
        let recognizer = params
            .recognition_model
            .map(TextRecognizer::from_model)
            .transpose()?;
        Ok(OcrEngine {
            detector,
            recognizer,
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
        if let Some(detector) = self.detector.as_ref() {
            detector.detect_words(input.image.view(), self.debug)
        } else {
            Err("Detection model not loaded".into())
        }
    }

    /// Detect text pixels in an image.
    ///
    /// Returns an (H, W) tensor indicating the probability of each pixel in the
    /// input being part of a text word. This is a low-level API that is useful
    /// for debugging purposes. Use [detect_words](OcrEngine::detect_words) for
    /// a higher-level API that returns oriented bounding boxes of words.
    pub fn detect_text_pixels(&self, input: &OcrInput) -> Result<NdTensor<f32, 2>, Box<dyn Error>> {
        if let Some(detector) = self.detector.as_ref() {
            detector.detect_text_pixels(input.image.view(), self.debug)
        } else {
            Err("Detection model not loaded".into())
        }
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
        if let Some(recognizer) = self.recognizer.as_ref() {
            recognizer.recognize_text_lines(
                input.image.view(),
                lines,
                RecognitionOpt {
                    debug: self.debug,
                    decode_method: self.decode_method,
                },
            )
        } else {
            Err("Recognition model not loaded".into())
        }
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

    use rten::model_builder::{ModelBuilder, OpType};
    use rten::ops::{MaxPool, Transpose};
    use rten::Dimension;
    use rten::Model;
    use rten_imageproc::{fill_rect, BoundingRect, Rect, RectF, RotatedRect};
    use rten_tensor::prelude::*;
    use rten_tensor::{NdTensor, Tensor};

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
            .slice_mut::<2, _>((.., 2, ..))
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
