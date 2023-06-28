use wasm_bindgen::prelude::*;

use wasnn::Model;
use wasnn_tensor::{TensorLayout, TensorView};

use crate::{OcrEngine as BaseOcrEngine, OcrEngineParams, OcrInput};

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
            debug: false,
        });
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

        if data.len() % pixels_per_chan != 0 || ![1, 3, 4].contains(&channels) {
            return Err("Expected image data length to be 1, 3 or 4x width * height".to_string());
        }

        let tensor = TensorView::from_data(&[height, width, channels], data)
            .permuted(&[2, 0, 1]) // HWC => CHW
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
            .permuted(&[1, 2, 0])
            .iter()
            .map(|x| ((x + 0.5) * 255.) as u8)
            .collect()
    }
}
