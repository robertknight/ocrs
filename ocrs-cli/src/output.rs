use rten_imageproc::{min_area_rect, BoundingRect, Painter, Point, PointF, Rgb, RotatedRect};
use rten_tensor::{NdTensor, NdTensorView};
use serde_json::json;

use ocrs::{TextItem, TextLine};

pub enum OutputFormat {
    /// Output a PNG image containing a copy of the input image annotated with
    /// text bounding boxes.
    Png,

    /// Output extracted plain text in reading order.
    Text,

    /// Output text and layout information in JSON format.
    Json,
}

/// Generate a JSON representation of detected word boxes in an image.
///
/// The format here matches that used by the layout model / evaluation tools.
fn word_boxes_json(
    img_path: &str,
    image_hw: [usize; 2],
    boxes: &[RotatedRect],
) -> serde_json::Value {
    let word_items: Vec<_> = boxes
        .iter()
        .map(|wr| {
            let br = wr.bounding_rect();
            let coords = [br.left(), br.top(), br.right(), br.bottom()].map(|c| c.round() as i32);
            json!({
                "coords": coords,
            })
        })
        .collect();

    let [height, width] = image_hw;

    json!({
        "url": img_path,
        "resolution": {
            "width": width,
            "height": height,
        },

        // nb. Since we haven't got layout analysis info here, we just put all
        // the words in one paragraph.
        "paragraphs": [{
            "words": serde_json::Value::Array(word_items),
        }]
    })
}

/// Input data for [format_json_output].
pub struct FormatJsonArgs<'a> {
    pub input_path: &'a str,
    pub input_hw: [usize; 2],

    /// Word bounding boxes returned by OCR engine.
    pub word_rects: &'a [RotatedRect],
}

/// Format OCR outputs as plain text.
pub fn format_text_output(text_lines: &[Option<TextLine>]) -> String {
    let lines: Vec<String> = text_lines
        .iter()
        .flatten()
        .map(|line| line.to_string())
        .collect();
    lines.join("\n")
}

/// Format OCR outputs as JSON.
pub fn format_json_output(args: FormatJsonArgs) -> String {
    let json_data = word_boxes_json(args.input_path, args.input_hw, args.word_rects);
    serde_json::to_string_pretty(&json_data).expect("JSON formatting failed")
}

/// Arguments for [generate_annotated_png].
pub struct GeneratePngArgs<'a> {
    /// Input image as a (channels, height, width) tensor.
    pub img: NdTensorView<'a, f32, 3>,

    /// Lines of text detected by OCR engine.
    pub line_rects: &'a [Vec<RotatedRect>],

    /// Lines of text recognized by OCR engine.
    pub text_lines: &'a [Option<TextLine>],
}

/// Annotate OCR input image with detected text.
pub fn generate_annotated_png(args: GeneratePngArgs) -> NdTensor<f32, 3> {
    let GeneratePngArgs {
        img,
        line_rects,
        text_lines,
    } = args;
    let mut annotated_img = img.to_tensor();
    let mut painter = Painter::new(annotated_img.view_mut());

    // Colors chosen from https://www.w3.org/wiki/CSS/Properties/color/keywords.
    //
    // Light colors for text detection outputs, darker colors for
    // corresponding text recognition outputs.
    const CORAL: Rgb = [255, 127, 80];
    const DARKSEAGREEN: Rgb = [143, 188, 143];
    const CORNFLOWERBLUE: Rgb = [100, 149, 237];

    const CRIMSON: Rgb = [220, 20, 60];
    const DARKGREEN: Rgb = [0, 100, 0];
    const DARKBLUE: Rgb = [0, 0, 139];

    const LIGHT_GRAY: Rgb = [200, 200, 200];

    let u8_to_f32 = |x: u8| x as f32 / 255.;
    let floor_point = |p: PointF| Point::from_yx(p.y as i32, p.x as i32);

    // Draw line bounding rects from layout analysis step.
    for line in line_rects.iter() {
        let line_points: Vec<_> = line
            .iter()
            .flat_map(|word_rect| word_rect.corners().into_iter())
            .collect();
        if let Some(line_rect) = min_area_rect(&line_points) {
            painter.set_stroke(LIGHT_GRAY.map(u8_to_f32));
            painter.draw_polygon(&line_rect.corners().map(floor_point));
        };
    }

    // Draw word bounding rects from text detection step, grouped by line.
    let colors = [CORAL, DARKSEAGREEN, CORNFLOWERBLUE];
    for (line, color) in line_rects.iter().zip(colors.into_iter().cycle()) {
        for word_rect in line {
            painter.set_stroke(color.map(u8_to_f32));
            painter.draw_polygon(&word_rect.corners().map(floor_point));
        }
    }

    // Draw word bounding rects from text recognition step. These may be
    // different as they are computed from the bounding boxes of recognized
    // characters.
    let colors = [CRIMSON, DARKGREEN, DARKBLUE];
    for (line, color) in text_lines.iter().zip(colors.into_iter().cycle()) {
        let Some(line) = line else {
            // Skip lines where recognition produced no output.
            continue;
        };
        for text_word in line.words() {
            painter.set_stroke(color.map(u8_to_f32));
            painter.draw_polygon(&text_word.rotated_rect().corners().map(floor_point));
        }
    }

    annotated_img
}

#[cfg(test)]
mod tests {
    use ocrs::{TextChar, TextItem, TextLine};
    use rten_imageproc::Rect;
    use rten_tensor::prelude::*;
    use rten_tensor::NdTensor;

    use super::{
        format_json_output, format_text_output, generate_annotated_png, FormatJsonArgs,
        GeneratePngArgs,
    };

    fn gen_text_chars(text: &str, width: i32) -> Vec<TextChar> {
        text.chars()
            .enumerate()
            .map(|(i, char)| TextChar {
                char,
                rect: Rect::from_tlhw(0, i as i32 * width, 25, width),
            })
            .collect()
    }

    #[test]
    fn test_format_json_output() {
        let line = TextLine::new(gen_text_chars("this is a test line", 10));
        let word_rects: Vec<_> = line.words().map(|w| w.rotated_rect()).collect();

        let json = format_json_output(FormatJsonArgs {
            input_path: "image.jpeg",
            input_hw: [256, 256],
            word_rects: &word_rects,
        });

        // At present this is just a trivial "does it run" test.
        assert!(!json.is_empty());
    }

    #[test]
    fn test_format_text_output() {
        let lines = &[
            Some(TextLine::new(gen_text_chars("line one", 10))),
            None,
            Some(TextLine::new(gen_text_chars("line two", 10))),
        ];
        let formatted = format_text_output(lines);
        let formatted_lines: Vec<_> = formatted.lines().collect();

        assert_eq!(formatted_lines, ["line one", "line two",]);
    }

    #[test]
    fn test_generate_annotated_png() {
        let img = NdTensor::zeros([3, 64, 64]);
        let text_lines = &[
            Some(TextLine::new(gen_text_chars("line one", 10))),
            Some(TextLine::new(gen_text_chars("line one", 10))),
        ];

        let line_rects: Vec<_> = text_lines
            .iter()
            .filter_map(|line| line.clone().map(|l| vec![l.rotated_rect()]))
            .collect();

        let args = GeneratePngArgs {
            img: img.view(),
            line_rects: &line_rects,
            text_lines,
        };

        let annotated = generate_annotated_png(args);

        assert_eq!(annotated.shape(), img.shape());
    }
}
