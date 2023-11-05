use wasm_bindgen::prelude::*;

use wasnn::ops;
use wasnn::{Model, OpRegistry};

use wasnn_imageproc::{min_area_rect, BoundingRect, PointF};
use wasnn_tensor::prelude::*;
use wasnn_tensor::NdTensorView;

use crate::{OcrEngine as BaseOcrEngine, OcrEngineParams, OcrInput, TextItem};

/// Create a collection type that can be used to accept or return collections
/// from / to JS. On the JS side, these collections are similar to DOM APIs
/// like [NodeList](https://developer.mozilla.org/en-US/docs/Web/API/NodeList)
/// and can be enhanced by implementing a `[Symbol.iterator]` method so they
/// work with `Array.from`, `for ... of` etc:
///
/// ```js
/// ThingList.prototype[Symbol.iterator] = function* () {
///   for (let i=0; i < this.length; i++) {
///     yield this.item(i);
///   }
/// }
///
/// const list = new ThingList();
/// list.push(new Thing());
///
/// for (let item of list) {
///
/// }
/// ```
///
/// These collections are a workaround for wasm_bindgen not supporting returning
/// `Vec<T>` to JS (see
/// [issue](https://github.com/rustwasm/wasm-bindgen/issues/111)).
macro_rules! make_item_list {
    ($list_struct:ident, Option<$item_struct:ty>) => {
        #[wasm_bindgen]
        #[derive(Clone, Default)]
        pub struct $list_struct {
            items: Vec<Option<$item_struct>>,
        }

        impl $list_struct {
            #[allow(dead_code)]
            fn from_vec(items: Vec<Option<$item_struct>>) -> Self {
                Self { items }
            }

            #[allow(dead_code)]
            fn iter(&self) -> impl Iterator<Item = &Option<$item_struct>> {
                self.items.iter()
            }
        }

        #[wasm_bindgen]
        impl $list_struct {
            #[wasm_bindgen(constructor)]
            pub fn new() -> Self {
                Self { items: Vec::new() }
            }

            /// Add a new item to the end of the list.
            pub fn push(&mut self, item: Option<$item_struct>) {
                self.items.push(item.clone());
            }

            /// Return the item at a given index.
            pub fn item(&self, index: usize) -> Option<$item_struct> {
                self.items.get(index).cloned().flatten()
            }

            #[wasm_bindgen(getter)]
            pub fn length(&self) -> usize {
                self.items.len()
            }
        }
    };

    ($list_struct:ident, $item_struct:ty) => {
        #[wasm_bindgen]
        #[derive(Clone, Default)]
        pub struct $list_struct {
            items: Vec<$item_struct>,
        }

        impl $list_struct {
            #[allow(dead_code)]
            fn from_vec(items: Vec<$item_struct>) -> Self {
                Self { items }
            }

            #[allow(dead_code)]
            fn iter(&self) -> impl Iterator<Item = &$item_struct> {
                self.items.iter()
            }
        }

        #[wasm_bindgen]
        impl $list_struct {
            #[wasm_bindgen(constructor)]
            pub fn new() -> Self {
                Self { items: Vec::new() }
            }

            /// Add a new item to the end of the list.
            pub fn push(&mut self, item: $item_struct) {
                self.items.push(item.clone());
            }

            /// Return the item at a given index.
            pub fn item(&self, index: usize) -> Option<$item_struct> {
                self.items.get(index).cloned()
            }

            #[wasm_bindgen(getter)]
            pub fn length(&self) -> usize {
                self.items.len()
            }
        }
    };
}

/// Options for constructing an [OcrEngine].
#[wasm_bindgen]
pub struct OcrEngineInit {
    op_registry: OpRegistry,
    detection_model: Option<Model>,
    recognition_model: Option<Model>,
}

impl Default for OcrEngineInit {
    fn default() -> OcrEngineInit {
        OcrEngineInit::new()
    }
}

#[wasm_bindgen]
impl OcrEngineInit {
    #[wasm_bindgen(constructor)]
    pub fn new() -> OcrEngineInit {
        let mut reg = OpRegistry::new();

        // Register all the operators the OCR models currently use.
        reg.register_op::<ops::Add>();
        reg.register_op::<ops::AveragePool>();
        reg.register_op::<ops::Cast>();
        reg.register_op::<ops::Concat>();
        reg.register_op::<ops::ConstantOfShape>();
        reg.register_op::<ops::Conv>();
        reg.register_op::<ops::ConvTranspose>();
        reg.register_op::<ops::GRU>();
        reg.register_op::<ops::Gather>();
        reg.register_op::<ops::LogSoftmax>();
        reg.register_op::<ops::MatMul>();
        reg.register_op::<ops::MaxPool>();
        reg.register_op::<ops::Pad>();
        reg.register_op::<ops::Relu>();
        reg.register_op::<ops::Reshape>();
        reg.register_op::<ops::Shape>();
        reg.register_op::<ops::Sigmoid>();
        reg.register_op::<ops::Slice>();
        reg.register_op::<ops::Transpose>();
        reg.register_op::<ops::Unsqueeze>();

        OcrEngineInit {
            op_registry: reg,
            detection_model: None,
            recognition_model: None,
        }
    }

    /// Load a model for text detection.
    #[wasm_bindgen(js_name = setDetectionModel)]
    pub fn set_detection_model(&mut self, data: &[u8]) -> Result<(), String> {
        let model = Model::load_with_ops(data, &self.op_registry)?;
        self.detection_model = Some(model);
        Ok(())
    }

    /// Load a model for text recognition.
    #[wasm_bindgen(js_name = setRecognitionModel)]
    pub fn set_recognition_model(&mut self, data: &[u8]) -> Result<(), String> {
        let model = Model::load_with_ops(data, &self.op_registry)?;
        self.recognition_model = Some(model);
        Ok(())
    }
}

/// OcrEngine is the main API for performing OCR in WebAssembly.
#[wasm_bindgen]
pub struct OcrEngine {
    engine: BaseOcrEngine,
}

#[wasm_bindgen]
impl OcrEngine {
    /// Construct a new `OcrEngine` using the models and other settings given
    /// by `init`.
    ///
    /// To detect text in an image, `init` must have a detection model set.
    /// To recognize text, `init` must have a recognition model set.
    #[wasm_bindgen(constructor)]
    pub fn new(init: OcrEngineInit) -> Result<OcrEngine, String> {
        let OcrEngineInit {
            detection_model,
            recognition_model,
            op_registry: _op_registry,
        } = init;
        let engine = BaseOcrEngine::new(OcrEngineParams {
            detection_model,
            recognition_model,
            ..Default::default()
        })
        .map_err(|e| e.to_string())?;
        Ok(OcrEngine { engine })
    }

    /// Prepare an image for analysis by the OCR engine.
    ///
    /// The image is an array of pixels in row-major, channels last order. This
    /// matches the format of the
    /// [ImageData](https://developer.mozilla.org/en-US/docs/Web/API/ImageData)
    /// API. Supported channel combinations are RGB and RGBA. The number of
    /// channels is inferred from the length of `data`.
    #[wasm_bindgen(js_name = loadImage)]
    pub fn load_image(&self, width: usize, height: usize, data: &[u8]) -> Result<Image, String> {
        let pixels_per_chan = height * width;
        let channels = data.len() / pixels_per_chan;

        if ![1, 3, 4].contains(&channels) {
            return Err("expected channel count to be 1, 3 or 4".to_string());
        }

        let tensor = NdTensorView::from_slice(data, [height, width, channels], None)
            .map_err(|_| "incorrect data length for image size and channel count".to_string())?
            .permuted([2, 0, 1]) // HWC => CHW
            .map(|x| (*x as f32) / 255.);
        self.engine
            .prepare_input(tensor.view())
            .map(|input| Image { input })
            .map_err(|e| e.to_string())
    }

    /// Detect text in an image.
    ///
    /// Returns a list of lines that were found. These can be passed to
    /// `recognizeText` identify the characters.
    #[wasm_bindgen(js_name = detectText)]
    pub fn detect_text(&self, image: &Image) -> Result<DetectedLineList, String> {
        let words = self
            .engine
            .detect_words(&image.input)
            .map_err(|e| e.to_string())?;
        let lines: Vec<_> = self
            .engine
            .find_text_lines(&image.input, &words)
            .into_iter()
            .map(|words| {
                DetectedLine::new(
                    words
                        .into_iter()
                        .map(|word| RotatedRect { rect: word })
                        .collect(),
                )
            })
            .collect();
        Ok(DetectedLineList::from_vec(lines))
    }

    /// Recognize text that was previously detected with `detectText`.
    ///
    /// Returns a list of `TextLine` objects that can be used to query the text
    /// and bounding boxes of each line.
    #[wasm_bindgen(js_name = recognizeText)]
    pub fn recognize_text(
        &self,
        image: &Image,
        lines: &DetectedLineList,
    ) -> Result<TextLineList, String> {
        let lines: Vec<Vec<wasnn_imageproc::RotatedRect>> = lines
            .iter()
            .map(|line| {
                let words: Vec<wasnn_imageproc::RotatedRect> =
                    line.words.iter().map(|word| word.rect).collect();
                words
            })
            .collect();

        let text_lines = self
            .engine
            .recognize_text(&image.input, &lines)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|line| line.map(|line| TextLine { line }))
            .collect();
        Ok(TextLineList::from_vec(text_lines))
    }

    /// Detect and recognize text in an image.
    ///
    /// Returns a single string containing all the text found in reading order.
    #[wasm_bindgen(js_name = getText)]
    pub fn get_text(&self, image: &Image) -> Result<String, String> {
        self.engine
            .get_text(&image.input)
            .map_err(|e| e.to_string())
    }

    /// Detect and recognize text in an image.
    ///
    /// Returns a list of `TextLine` objects that can be used to query the text
    /// and bounding boxes of each line.
    #[wasm_bindgen(js_name = getTextLines)]
    pub fn get_text_lines(&self, image: &Image) -> Result<TextLineList, String> {
        let words = self
            .engine
            .detect_words(&image.input)
            .map_err(|e| e.to_string())?;
        let lines = self.engine.find_text_lines(&image.input, &words);
        let text_lines = self
            .engine
            .recognize_text(&image.input, &lines)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|line| line.map(|line| TextLine { line }))
            .collect();
        Ok(TextLineList::from_vec(text_lines))
    }
}

/// A pre-processed image that can be passed as input to `OcrEngine.loadImage`.
#[wasm_bindgen]
pub struct Image {
    input: OcrInput,
}

#[wasm_bindgen]
impl Image {
    /// Return the number of channels in the image.
    pub fn channels(&self) -> usize {
        self.input.image.size(0)
    }

    /// Return the width of the image.
    pub fn width(&self) -> usize {
        self.input.image.size(2)
    }

    /// Return the height of the image.
    pub fn height(&self) -> usize {
        self.input.image.size(1)
    }

    /// Return the image data in row-major, channels-last order.
    pub fn data(&self) -> Vec<u8> {
        // Permute CHW => HWC, convert pixel values from [-0.5, 0.5] back to
        // [0, 255].
        self.input
            .image
            .permuted([1, 2, 0])
            .iter()
            .map(|x| ((x + 0.5) * 255.) as u8)
            .collect()
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct RotatedRect {
    rect: wasnn_imageproc::RotatedRect,
}

#[wasm_bindgen]
impl RotatedRect {
    /// Return an array of the X and Y coordinates of corners of this rectangle,
    /// arranged as `[x0, y0, ... x3, y3]`.
    pub fn corners(&self) -> Vec<f32> {
        self.rect
            .corners()
            .into_iter()
            .flat_map(|c| [c.x, c.y])
            .collect()
    }

    /// Return the coordinates of the axis-aligned bounding rectangle of this
    /// rotated rect.
    ///
    /// The result is a `[left, top, right, bottom]` array of coordinates.
    #[wasm_bindgen(js_name = boundingRect)]
    pub fn bounding_rect(&self) -> Vec<f32> {
        let br = self.rect.bounding_rect();
        [br.left(), br.top(), br.right(), br.bottom()].into()
    }
}

make_item_list!(RotatedRectList, RotatedRect);

/// A line of text that has been detected, but not recognized.
///
/// This contains information about the location of the text, but not the
/// string contents.
#[wasm_bindgen]
#[derive(Clone)]
pub struct DetectedLine {
    words: RotatedRectList,
}

#[wasm_bindgen]
impl DetectedLine {
    fn new(words: Vec<RotatedRect>) -> DetectedLine {
        DetectedLine {
            words: RotatedRectList::from_vec(words),
        }
    }

    #[wasm_bindgen(js_name = rotatedRect)]
    pub fn rotated_rect(&self) -> RotatedRect {
        let points: Vec<PointF> = self
            .words
            .iter()
            .flat_map(|word| word.rect.corners().into_iter())
            .collect();
        let rect = min_area_rect(&points).expect("expected non-empty rect");
        RotatedRect { rect }
    }

    pub fn words(&self) -> RotatedRectList {
        self.words.clone()
    }
}

make_item_list!(DetectedLineList, DetectedLine);

/// Bounding box and text of a word that was recognized.
#[wasm_bindgen]
#[derive(Clone)]
pub struct TextWord {
    rect: RotatedRect,
    text: String,
}

#[wasm_bindgen]
impl TextWord {
    pub fn text(&self) -> String {
        self.text.clone()
    }

    /// Return the oriented bounding rectangle containing the characters in
    /// this word.
    #[wasm_bindgen(js_name = rotatedRect)]
    pub fn rotated_rect(&self) -> RotatedRect {
        self.rect.clone()
    }
}

make_item_list!(TextWordList, TextWord);

/// A sequence of `TextWord`s that were recognized, forming a line.
#[wasm_bindgen]
#[derive(Clone)]
pub struct TextLine {
    // Text line, or None if no text was recognized.
    line: super::TextLine,
}

#[wasm_bindgen]
impl TextLine {
    pub fn text(&self) -> String {
        self.line.to_string()
    }

    pub fn words(&self) -> TextWordList {
        let items: Vec<TextWord> = self
            .line
            .words()
            .map(|w| TextWord {
                text: w.to_string(),
                rect: RotatedRect {
                    rect: w.rotated_rect(),
                },
            })
            .collect();
        TextWordList::from_vec(items)
    }
}

make_item_list!(TextLineList, Option<TextLine>);
