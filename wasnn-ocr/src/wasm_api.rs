use wasm_bindgen::prelude::*;

use wasnn::Model;
use wasnn_imageproc::BoundingRect;
use wasnn_tensor::{Layout, NdTensorView};

use crate::{OcrEngine as BaseOcrEngine, OcrEngineParams, OcrInput, TextItem};

/// Options for constructing an [OcrEngine].
#[derive(Default)]
#[wasm_bindgen]
pub struct OcrEngineInit {
    detection_model: Option<Model>,
    recognition_model: Option<Model>,
}

#[wasm_bindgen]
impl OcrEngineInit {
    #[wasm_bindgen(constructor)]
    pub fn new() -> OcrEngineInit {
        OcrEngineInit {
            detection_model: None,
            recognition_model: None,
        }
    }

    /// Load a model for text detection.
    #[wasm_bindgen(js_name = setDetectionModel)]
    pub fn set_detection_model(&mut self, data: &[u8]) -> Result<(), String> {
        let model = Model::load(data)?;
        self.detection_model = Some(model);
        Ok(())
    }

    /// Load a model for text recognition.
    #[wasm_bindgen(js_name = setRecognitionModel)]
    pub fn set_recognition_model(&mut self, data: &[u8]) -> Result<(), String> {
        let model = Model::load(data)?;
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
            .filter_map(|text_line| text_line.map(|tl| TextLine { line: tl }))
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
            .view()
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
    pub fn corners(&self) -> Vec<i32> {
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
    pub fn bounding_rect(&self) -> Vec<i32> {
        let br = self.rect.bounding_rect();
        [br.left(), br.top(), br.right(), br.bottom()].into()
    }
}

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
    ($list_struct:ident, $item_struct:ident) => {
        #[wasm_bindgen]
        #[derive(Default)]
        pub struct $list_struct {
            items: Vec<$item_struct>,
        }

        #[wasm_bindgen]
        impl $list_struct {
            fn from_vec(items: Vec<$item_struct>) -> Self {
                Self { items }
            }

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

#[wasm_bindgen]
#[derive(Clone)]
pub struct TextLine {
    line: super::TextLine,
}

#[wasm_bindgen]
impl TextLine {
    pub fn text(&self) -> String {
        self.line.to_string()
    }

    pub fn words(&self) -> TextWordList {
        let items = self
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

make_item_list!(TextLineList, TextLine);
