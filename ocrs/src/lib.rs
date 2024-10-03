use anyhow::anyhow;
use rten::Model;
use rten_imageproc::RotatedRect;
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;

mod detection;
mod errors;
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

use detection::{TextDetector, TextDetectorParams};
use layout_analysis::find_text_lines;
use preprocess::prepare_image;
use recognition::{RecognitionOpt, TextRecognizer};

pub use preprocess::{DimOrder, ImagePixels, ImageSource, ImageSourceError};
pub use recognition::DecodeMethod;
pub use text_items::{TextChar, TextItem, TextLine, TextWord};

// nb. The "E" before "ABCDE" should be the EUR symbol.
const DEFAULT_ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~EABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Configuration for an [OcrEngine] instance.
#[derive(Default)]
pub struct OcrEngineParams {
    /// Model used to detect text words in the image.
    pub detection_model: Option<Model>,

    /// Model used to recognize lines of text in the image.
    ///
    /// If using a custom model, you may need to adjust the
    /// [`alphabet`](Self::alphabet) to match.
    pub recognition_model: Option<Model>,

    /// Enable debug logging.
    pub debug: bool,

    /// Method used to decode outputs of text recognition model.
    pub decode_method: DecodeMethod,

    /// Alphabet used for text recognition.
    ///
    /// This is useful if you are using a custom recognition model with a
    /// modified alphabet. If not specified a default alphabet will be used
    /// which matches the one used to train the [original
    /// models](https://github.com/robertknight/ocrs-models).
    pub alphabet: Option<String>,

    /// Set of characters that may be produced by text recognition.
    ///
    /// This is useful when you need the text recognition model to
    /// produce text that only includes a predefined set of characters, for
    /// example only numbers or lower-case letters.
    ///
    /// If this option is not set, text recognition may produce any character
    /// from the recognition model's alphabet.
    pub allowed_chars: Option<String>,
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
    alphabet: String,

    /// Indices of characters in `alphabet` that are excluded from recognition
    /// output. See [`OcrEngineParams::allowed_chars`].
    excluded_char_labels: Option<Vec<usize>>,
}

/// Input image for OCR analysis. Instances are created using
/// [OcrEngine::prepare_input]
pub struct OcrInput {
    /// CHW tensor with normalized pixel values in [BLACK_VALUE, BLACK_VALUE + 1.].
    pub(crate) image: NdTensor<f32, 3>,
}

impl OcrEngine {
    /// Construct a new engine from a given configuration.
    pub fn new(params: OcrEngineParams) -> anyhow::Result<OcrEngine> {
        let detector = params
            .detection_model
            .map(|model| TextDetector::from_model(model, Default::default()))
            .transpose()?;
        let recognizer = params
            .recognition_model
            .map(TextRecognizer::from_model)
            .transpose()?;

        let alphabet = params
            .alphabet
            .unwrap_or_else(|| DEFAULT_ALPHABET.to_string());

        let excluded_char_labels = params.allowed_chars.map(|allowed_characters| {
            alphabet
                .chars()
                .enumerate()
                .filter_map(|(index, char)| {
                    if !allowed_characters.contains(char) {
                        // Index `0` is reserved for the CTC blank character and
                        // `i + 1` is used as training label for character at
                        // index `i` of `alphabet` string.
                        //
                        // See https://github.com/robertknight/ocrs-models/blob/3d98fc655d6fd4acddc06e7f5d60a55b55748a48/ocrs_models/datasets/util.py#L113
                        Some(index + 1)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        });

        Ok(OcrEngine {
            detector,
            recognizer,
            alphabet,
            excluded_char_labels,
            debug: params.debug,
            decode_method: params.decode_method,
        })
    }

    /// Preprocess an image for use with other methods of the engine.
    pub fn prepare_input(&self, image: ImageSource) -> anyhow::Result<OcrInput> {
        Ok(OcrInput {
            image: prepare_image(image),
        })
    }

    /// Detect text words in an image.
    ///
    /// Returns an unordered list of the oriented bounding rectangles of each
    /// word found.
    pub fn detect_words(&self, input: &OcrInput) -> anyhow::Result<Vec<RotatedRect>> {
        if let Some(detector) = self.detector.as_ref() {
            detector.detect_words(input.image.view(), self.debug)
        } else {
            Err(anyhow!("Detection model not loaded"))
        }
    }

    /// Detect text pixels in an image.
    ///
    /// Returns an (H, W) tensor indicating the probability of each pixel in the
    /// input being part of a text word. This is a low-level API that is useful
    /// for debugging purposes. Use [detect_words](OcrEngine::detect_words) for
    /// a higher-level API that returns oriented bounding boxes of words.
    pub fn detect_text_pixels(&self, input: &OcrInput) -> anyhow::Result<NdTensor<f32, 2>> {
        if let Some(detector) = self.detector.as_ref() {
            detector.detect_text_pixels(input.image.view(), self.debug)
        } else {
            Err(anyhow!("Detection model not loaded"))
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
    ) -> anyhow::Result<Vec<Option<TextLine>>> {
        if let Some(recognizer) = self.recognizer.as_ref() {
            recognizer.recognize_text_lines(
                input.image.view(),
                lines,
                RecognitionOpt {
                    debug: self.debug,
                    decode_method: self.decode_method,
                    alphabet: &self.alphabet,
                    excluded_char_labels: self.excluded_char_labels.as_deref(),
                },
            )
        } else {
            Err(anyhow!("Recognition model not loaded"))
        }
    }

    /// Prepare an image for input into the text line recognition model.
    ///
    /// This method exists to help with debugging recognition issues by exposing
    /// the preprocessing that [OcrEngine::recognize_text] does before it feeds
    /// an image into the recognition model. Use [OcrEngine::recognize_text] to
    /// recognize text.
    ///
    /// `line` is a sequence of [RotatedRect]s that make up a line of text.
    ///
    /// Returns a greyscale (H, W) image with values in [-0.5, 0.5].
    pub fn prepare_recognition_input(
        &self,
        input: &OcrInput,
        line: &[RotatedRect],
    ) -> anyhow::Result<NdTensor<f32, 2>> {
        let Some(recognizer) = self.recognizer.as_ref() else {
            return Err(anyhow!("Recognition model not loaded"));
        };
        let line_image = recognizer.prepare_input(input.image.view(), line);
        Ok(line_image)
    }

    /// Return the confidence threshold applied to the output of the text
    /// detection model to determine whether a pixel is text or not.
    pub fn detection_threshold(&self) -> f32 {
        self.detector
            .as_ref()
            .map(|detector| detector.threshold())
            .unwrap_or(TextDetectorParams::default().text_threshold)
    }

    /// Convenience API that extracts all text from an image as a single string.
    pub fn get_text(&self, input: &OcrInput) -> anyhow::Result<String> {
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

    use rten::model_builder::{ModelBuilder, ModelFormat, OpType};
    use rten::ops::{MaxPool, Transpose};
    use rten::Dimension;
    use rten::Model;
    use rten_imageproc::{fill_rect, BoundingRect, Rect, RectF, RotatedRect};
    use rten_tensor::prelude::*;
    use rten_tensor::{NdTensor, NdTensorView, Tensor};

    use super::{DimOrder, ImageSource, OcrEngine, OcrEngineParams, DEFAULT_ALPHABET};

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
        let mut mb = ModelBuilder::new(ModelFormat::V1);
        let mut gb = mb.graph_builder();

        let input_id = gb.add_value(
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
        gb.add_input(input_id);

        let output_id = gb.add_value("output", None);
        gb.add_output(output_id);

        let bias = Tensor::from_scalar(0.5);
        let bias_id = gb.add_constant(bias.view());
        gb.add_operator(
            "add",
            OpType::Add,
            &[Some(input_id), Some(bias_id)],
            &[output_id],
        );

        let graph = gb.finish();
        mb.set_graph(graph);

        let model_data = mb.finish();
        Model::load(model_data).unwrap()
    }

    /// Create a fake text recognition model.
    ///
    /// This takes an NCHW input with C=1, H=64 and returns an output with
    /// shape `[W / 4, N, C]`. In the real model the last dimension is the
    /// log-probability of each class label. In this fake we just re-interpret
    /// each column of the input as a one-hot vector of probabilities.
    ///
    /// Returns a `(model, alphabet)` tuple.
    fn fake_recognition_model() -> (Model, String) {
        let mut mb = ModelBuilder::new(ModelFormat::V1);
        let mut gb = mb.graph_builder();

        let output_columns = 64;
        let input_id = gb.add_value(
            "input",
            Some(&[
                Dimension::Symbolic("batch".to_string()),
                Dimension::Fixed(1),
                Dimension::Fixed(output_columns),
                Dimension::Symbolic("seq".to_string()),
            ]),
        );
        gb.add_input(input_id);

        // MaxPool to scale width by 1/4: NCHW => NCHW/4
        let pool_out = gb.add_value("max_pool_out", None);
        gb.add_operator(
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
        let squeeze_axes_id = gb.add_constant(squeeze_axes.view());
        let squeeze_out = gb.add_value("squeeze_out", None);
        gb.add_operator(
            "squeeze",
            OpType::Squeeze,
            &[Some(pool_out), Some(squeeze_axes_id)],
            &[squeeze_out],
        );

        // Transpose: NHW/4 => W/4NH
        let transpose_out = gb.add_value("transpose_out", None);
        gb.add_operator(
            "transpose",
            OpType::Transpose(Transpose {
                perm: Some(vec![2, 0, 1]),
            }),
            &[Some(squeeze_out)],
            &[transpose_out],
        );

        gb.add_output(transpose_out);
        let graph = gb.finish();

        mb.set_graph(graph);

        let model_data = mb.finish();
        let model = Model::load(model_data).unwrap();
        let alphabet = DEFAULT_ALPHABET.chars().take(output_columns - 1).collect();

        (model, alphabet)
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
        let input = engine.prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;

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
        let input = engine.prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;
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

    // Test recognition using a dummy recognition model.
    //
    // The dummy model treats each column of the input image as a one-hot vector
    // of character class probabilities. Pre-processing of the input will shift
    // values from [0, 1] to [-0.5, 0.5]. CTC decoding of the output will ignore
    // class 0 (as it represents a CTC blank) and repeated characters.
    //
    // Filling a single input row with "1"s will produce a single char output
    // where the char's index in the alphabet is the row index - 1.  ie. Filling
    // the first row produces " ", the second row "0" and so on, using the
    // default alphabet.
    fn test_recognition(
        params: OcrEngineParams,
        image: NdTensorView<f32, 3>,
        expected_text: &str,
    ) -> Result<(), Box<dyn Error>> {
        let engine = OcrEngine::new(params)?;
        let input = engine.prepare_input(ImageSource::from_tensor(image.view(), DimOrder::Chw)?)?;

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
        assert_eq!(line.to_string(), expected_text);

        Ok(())
    }

    #[test]
    fn test_ocr_engine_recognize_lines() -> Result<(), Box<dyn Error>> {
        let mut image = NdTensor::zeros([1, 64, 32]);

        // Set the probability of character 1 in the alphabet ('0') to 1 and
        // leave all other characters with a probability of zero.
        image.slice_mut::<2, _>((.., 2, ..)).fill(1.);

        let (rec_model, alphabet) = fake_recognition_model();
        test_recognition(
            OcrEngineParams {
                detection_model: None,
                recognition_model: Some(rec_model),
                alphabet: Some(alphabet),
                ..Default::default()
            },
            image.view(),
            "0",
        )?;

        Ok(())
    }

    #[test]
    fn test_ocr_engine_filter_chars() -> Result<(), Box<dyn Error>> {
        let mut image = NdTensor::zeros([1, 64, 32]);

        // Set the probability of "0" to 0.7 and "1" to 0.3.
        image.slice_mut::<2, _>((.., 2, ..)).fill(0.7);
        image.slice_mut::<2, _>((.., 3, ..)).fill(0.3);

        let (rec_model, alphabet) = fake_recognition_model();
        test_recognition(
            OcrEngineParams {
                detection_model: None,
                recognition_model: Some(rec_model),
                alphabet: Some(alphabet),
                ..Default::default()
            },
            image.view(),
            "0",
        )?;

        // Run recognition again but exclude "0" from the output.
        let (rec_model, alphabet) = fake_recognition_model();
        test_recognition(
            OcrEngineParams {
                detection_model: None,
                recognition_model: Some(rec_model),
                alphabet: Some(alphabet),
                allowed_chars: Some("123456789".into()),
                ..Default::default()
            },
            image.view(),
            "1",
        )?;

        Ok(())
    }
}
