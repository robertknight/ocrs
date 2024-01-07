use rten_imageproc::{min_area_rect, Painter, Point, PointF, Rgb, RotatedRect};
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

/// Return the coordinates of vertices of `rr` as an array of `[x, y]` points.
///
/// This matches the format of the "vertices" data in the HierText dataset.
/// See [RotatedRect::corners] for details of the vertex order.
fn rounded_vertex_coords(rr: &RotatedRect) -> [[i32; 2]; 4] {
    rr.corners()
        .map(|point| [point.x.round() as i32, point.y.round() as i32])
}

/// Format extracted text and hierarchical layout information as JSON.
///
/// The JSON format roughly follows the structure of the ground truth data in
/// the [HierText](https://github.com/google-research-datasets/hiertext)
/// dataset, on which ocrs's models were trained.
fn ocr_json(args: FormatJsonArgs) -> serde_json::Value {
    let FormatJsonArgs {
        input_path,
        input_hw,
        text_lines,
    } = args;

    let line_items: Vec<_> = text_lines
        .iter()
        .filter_map(|line| line.as_ref())
        .map(|line| {
            let word_items: Vec<_> = line
                .words()
                .map(|word| {
                    json!({
                        "text": word.to_string(),
                        "vertices": rounded_vertex_coords(&word.rotated_rect()),
                    })
                })
                .collect();

            json!({
                "text": line.to_string(),
                "words": word_items,
                "vertices": rounded_vertex_coords(&line.rotated_rect()),
            })
        })
        .collect();

    let [height, width] = input_hw;

    json!({
        "url": input_path,
        "image_width": width,
        "image_height": height,

        // nb. Since we haven't got layout analysis info here, we just put all
        // the lines on one paragraph.
        "paragraphs": [{
            "lines": serde_json::Value::Array(line_items),
        }]
    })
}

/// Input data for [format_json_output].
pub struct FormatJsonArgs<'a> {
    pub input_path: &'a str,
    pub input_hw: [usize; 2],

    /// Lines of text recognized by OCR engine.
    pub text_lines: &'a [Option<TextLine>],
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
    let json_data = ocr_json(args);
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
    for line in line_rects {
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
    use std::fs::read_to_string;
    use std::io;
    use std::path::PathBuf;

    use ocrs::{TextChar, TextItem, TextLine};
    use rten_imageproc::Rect;
    use rten_tensor::prelude::*;
    use rten_tensor::NdTensor;

    use super::{
        format_json_output, format_text_output, generate_annotated_png, FormatJsonArgs,
        GeneratePngArgs,
    };

    /// Generate dummy OCR output with the given text and character spacing.
    fn gen_text_chars(text: &str, width: i32) -> Vec<TextChar> {
        text.chars()
            .enumerate()
            .map(|(i, char)| TextChar {
                char,
                rect: Rect::from_tlhw(0, i as i32 * width, 25, width),
            })
            .collect()
    }

    fn read_test_file(path: &str) -> Result<String, io::Error> {
        let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        abs_path.push("test-data/");
        abs_path.push(path);
        read_to_string(abs_path)
    }

    #[test]
    fn test_format_json_output() {
        let lines = &[
            Some(TextLine::new(gen_text_chars("line one", 10))),
            None,
            Some(TextLine::new(gen_text_chars("line two", 10))),
        ];

        let json = format_json_output(FormatJsonArgs {
            input_path: "image.jpeg",
            input_hw: [256, 256],
            text_lines: lines,
        });
        let parsed_json: serde_json::Value = serde_json::from_str(&json).unwrap();

        let expected_json = read_test_file("format-json-expected.json").unwrap();
        let expected: serde_json::Value = serde_json::from_str(&expected_json).unwrap();
        assert_eq!(parsed_json, expected);
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
