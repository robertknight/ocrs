use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io::BufWriter;

use rayon::prelude::*;

use wasnn::ctc::{CtcDecoder, CtcHypothesis};
use wasnn::ops::{pad, resize, CoordTransformMode, NearestMode, ResizeMode, ResizeTarget};
use wasnn::{Dimension, Model, RunOptions};
use wasnn_imageproc::{draw_polygon, Point, Polygon, Rect, RotatedRect};
use wasnn_ocr::page_layout::{find_connected_component_rects, find_text_lines, line_polygon};
use wasnn_tensor::{tensor, Tensor, TensorLayout, TensorView};

/// Read an image from `path` into a CHW tensor.
fn read_image(path: &str) -> Result<Tensor<f32>, Box<dyn Error>> {
    let input_img = image::open(path)?;
    let input_img = input_img.into_rgb8();

    let (width, height) = input_img.dimensions();

    let in_chans = 3;
    let mut float_img = Tensor::zeros(&[in_chans, height as usize, width as usize]);
    for c in 0..in_chans {
        let mut chan_img = float_img.nd_slice_mut([c]);
        for y in 0..height {
            for x in 0..width {
                chan_img[[y as usize, x as usize]] = input_img.get_pixel(x, y)[c] as f32 / 255.0
            }
        }
    }
    Ok(float_img)
}

/// Convert a CHW image into a greyscale image.
///
/// This function is intended to approximately match torchvision's RGB =>
/// greyscale conversion when using `torchvision.io.read_image(path,
/// ImageReadMode.GRAY)`, which is used when training models with greyscale
/// inputs. torchvision internally uses libpng's `png_set_rgb_to_gray`.
///
/// `normalize_pixel` is a function applied to each greyscale pixel value before
/// it is written into the output tensor.
fn greyscale_image<F: Fn(f32) -> f32>(img: TensorView<f32>, normalize_pixel: F) -> Tensor<f32> {
    let [chans, height, width]: [usize; 3] = img.shape().try_into().expect("expected 3 dim input");
    assert!(
        chans == 1 || chans == 3 || chans == 4,
        "expected greyscale, RGB or RGBA input image"
    );

    let mut output = Tensor::zeros(&[1, height, width]);

    let used_chans = chans.min(3); // For RGBA images, only RGB channels are used
    let chan_weights: &[f32] = if chans == 1 {
        &[1.]
    } else {
        // ITU BT.601 weights for RGB => luminance conversion. These match what
        // torchvision uses. See also https://stackoverflow.com/a/596241/434243.
        &[0.299, 0.587, 0.114]
    };

    let img = img.nd_view();
    let mut out_lum_chan = output.nd_slice_mut([0]);

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

/// Write a CHW image to a PNG file in `path`.
fn write_image(path: &str, img: TensorView<f32>) -> Result<(), Box<dyn Error>> {
    if img.ndim() != 3 {
        return Err("Expected CHW input".into());
    }

    let img_width = img.shape()[img.ndim() - 1];
    let img_height = img.shape()[img.ndim() - 2];
    let color_type = match img.shape()[img.ndim() - 3] {
        1 => png::ColorType::Grayscale,
        3 => png::ColorType::Rgb,
        4 => png::ColorType::Rgba,
        _ => return Err("Unsupported channel count".into()),
    };

    let mut hwc_img = img.to_owned();
    hwc_img.permute(&[1, 2, 0]); // CHW => HWC

    let out_img = image_from_tensor(hwc_img);
    let file = fs::File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, img_width as u32, img_height as u32);
    encoder.set_color(color_type);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&out_img)?;

    Ok(())
}

/// Convert an NCHW float tensor with values in the range [0, 1] to Vec<u8>
/// with values scaled to [0, 255].
fn image_from_tensor(tensor: TensorView<f32>) -> Vec<u8> {
    tensor
        .iter()
        .map(|x| (x.clamp(0., 1.) * 255.0) as u8)
        .collect()
}

// nb. The "E" before "ABCDE" should be the EUR symbol.
const DEFAULT_ALPHABET: &str = " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~EABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Return the smallest multiple of `factor` that is >= `val`.
fn round_up<
    T: Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Sub<T, Output = T>
        + std::ops::Rem<T, Output = T>,
>(
    val: T,
    factor: T,
) -> T {
    let rem = val % factor;
    (val + factor) - rem
}

/// Calculate the bounding rect of `rects`.
fn bounding_rect(rects: &[RotatedRect]) -> Option<Rect> {
    if let Some((min_x, max_x, min_y, max_y)) =
        rects
            .iter()
            .fold(None as Option<(i32, i32, i32, i32)>, |min_max, rect| {
                let br = rect.bounding_rect();
                min_max
                    .map(|(min_x, max_x, min_y, max_y)| {
                        (
                            min_x.min(br.left()),
                            max_x.max(br.right()),
                            min_y.min(br.top()),
                            max_y.max(br.bottom()),
                        )
                    })
                    .or(Some((br.left(), br.right(), br.top(), br.bottom())))
            })
    {
        Some(Rect::from_tlbr(min_y, min_x, max_y, max_x))
    } else {
        None
    }
}

/// The value used to represent fully black pixels in normalized greyscale images.
const ZERO_VALUE: f32 = -0.5;

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
fn detect_words(
    mut image: TensorView<f32>,
    model: &Model,
    debug: bool,
) -> Result<Vec<RotatedRect>, Box<dyn Error>> {
    let input_id = model
        .input_ids()
        .first()
        .copied()
        .expect("model has no inputs");
    let input_shape = model
        .node_info(input_id)
        .and_then(|info| info.shape())
        .ok_or("model does not specify expected input shape")?;
    let output_id = model
        .output_ids()
        .first()
        .copied()
        .expect("model has no outputs");

    let [_, img_height, img_width] = image.dims();
    image.insert_dim(0); // Add batch dimension

    let bilinear_resize = |img: TensorView, height, width| {
        resize(
            img,
            ResizeTarget::Sizes(&tensor!([1, 1, height, width])),
            ResizeMode::Linear,
            CoordTransformMode::default(),
            NearestMode::default(),
        )
    };

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
        pad(
            image.view(),
            &tensor!([0, 0, 0, 0, 0, 0, pad_bottom, pad_right]),
            ZERO_VALUE,
        )?
    } else {
        image.to_tensor()
    };

    // Resize images to the text detection model's input size.
    let resized_grey_img = bilinear_resize(grey_img.view(), in_height as i32, in_width as i32)?;

    // Run text detection model to compute a probability mask indicating whether
    // each pixel is part of a text word or not.
    let outputs = model.run(
        &[(input_id, (&resized_grey_img).into())],
        &[output_id],
        if debug {
            Some(RunOptions {
                timing: true,
                verbose: false,
            })
        } else {
            None
        },
    )?;

    // Resize probability mask to original input size and apply threshold to get a
    // binary text/not-text mask.
    let text_mask = &outputs[0].as_float_ref().unwrap();
    let text_mask = bilinear_resize(
        text_mask.slice((
            ..,
            ..,
            0..(in_height - pad_bottom as usize),
            0..(in_width - pad_right as usize),
        )),
        img_height as i32,
        img_width as i32,
    )?;
    let threshold = 0.2;
    let binary_mask = text_mask.map(|prob| if prob > threshold { 1i32 } else { 0 });

    // Distance to expand bounding boxes by. This is useful when the model is
    // trained to assign a positive label to pixels in a smaller area than the
    // ground truth, which may be done to create separation between adjacent
    // objects.
    let expand_dist = 3.;

    let word_rects = find_connected_component_rects(binary_mask.nd_slice([0, 0]), expand_dist);

    Ok(word_rects)
}

/// Recognize text lines in an image.
///
/// `image` is a CHW greyscale image with values in the range `ZERO_VALUE` to
/// `ZERO_VALUE + 1`. `lines` is a list of detected text lines, where each line
/// is a sequence of word rects. `model` is a recognition model which accepts an
/// NCHW tensor of greyscale line images and outputs a `[sequence, batch, label]`
/// tensor of log probabilities of character classes, which must be converted to
/// a character sequence using CTC decoding.
///
/// The result is a list containing the recognized text for each input line.
fn recognize_text_lines(
    image: TensorView<f32>,
    lines: &[Vec<RotatedRect>],
    page_rect: Rect,
    model: &Model,
    debug: bool,
) -> Result<Vec<String>, Box<dyn Error>> {
    let rec_input_id = model
        .input_ids()
        .first()
        .copied()
        .expect("recognition model has no inputs");
    let rec_input_shape = model
        .node_info(rec_input_id)
        .and_then(|info| info.shape())
        .ok_or("recognition model does not specify input shape")?;
    let rec_output_id = model
        .output_ids()
        .first()
        .copied()
        .ok_or("recognition model has no outputs")?;

    let rec_img_height = match rec_input_shape[2] {
        Dimension::Fixed(size) => size,
        Dimension::Symbolic(_) => 50,
    };

    // Compute sizes of line images after resizing to fixed height and scaled
    // width.
    let resized_line_width = |orig_width, orig_height| {
        // Min/max widths for resized line images. These must match the PyTorch
        // `HierTextRecognition` dataset loader.
        let min_width = 10.;
        let max_width = 800.;
        let aspect_ratio = orig_width as f32 / orig_height as f32;
        (rec_img_height as f32 * aspect_ratio)
            .max(min_width)
            .min(max_width) as i32
    };

    // Group lines into batches which will have similar widths after resizing
    // to a fixed height.
    //
    // It is more efficient to run recognition on multiple lines at once, but
    // all line images in a batch must be padded to an equal length. Some
    // computation is wasted on shorter lines in the batch. Choosing batches
    // such that all line images have a similar width reduces this wastage.
    // There is a trade-off between maximizing the batch size and minimizing
    // the variance in width of images in the batch.
    let mut line_groups: HashMap<i32, Vec<usize>> = HashMap::new();
    for (line_index, word_rects) in lines.iter().enumerate() {
        let line_rect = bounding_rect(word_rects).expect("line has no words");
        let resized_width = resized_line_width(line_rect.width(), line_rect.height());
        let group_width = round_up(resized_width, 50);
        line_groups.entry(group_width).or_default().push(line_index);
    }

    // Split large line groups up to better exploit parallelism in the loop
    // below.
    let max_lines_per_group = 20;
    let line_groups: Vec<(i32, Vec<usize>)> = line_groups
        .into_iter()
        .flat_map(|(group_width, line_indices)| {
            line_indices
                .chunks(max_lines_per_group)
                .map(move |chunk| (group_width, chunk.to_vec()))
                .collect::<Vec<_>>()
        })
        .collect();

    // Run text recognition on batches of lines. Produces a map of line index
    // to recognition result.
    let line_rec_results: HashMap<usize, CtcHypothesis> = line_groups
        .par_iter()
        .flat_map(|(group_width, line_indices)| {
            if debug {
                println!(
                    "Processing group of {} lines of width {}",
                    line_indices.len(),
                    group_width,
                );
            }

            let mut line_rec_input =
                Tensor::zeros(&[line_indices.len(), 1, rec_img_height, *group_width as usize]);

            // Extract and resize line images
            for (group_line_index, line_index) in line_indices.iter().copied().enumerate() {
                let word_rects = &lines[line_index];
                let line_poly = Polygon::new(line_polygon(word_rects));
                let grey_chan = image.nd_slice([0]);

                // Extract line image
                let line_rect = line_poly.bounding_rect();
                let mut out_img = Tensor::zeros(&[
                    1,
                    1,
                    line_rect.height() as usize,
                    line_rect.width() as usize,
                ]);
                out_img.iter_mut().for_each(|el| *el = 0.5);

                // Page rect adjusted to only contain coordinates that are valid for
                // indexing into the input image.
                let page_index_rect = page_rect.adjust_tlbr(0, 0, -1, -1);

                let mut out_hw = out_img.nd_slice_mut([0, 0]);
                for in_p in line_poly.fill_iter() {
                    let out_p = Point::from_yx(in_p.y - line_rect.top(), in_p.x - line_rect.left());

                    if !page_index_rect.contains_point(in_p)
                        || !page_index_rect.contains_point(out_p)
                    {
                        continue;
                    }

                    let normalized_pixel = grey_chan[[in_p.y as usize, in_p.x as usize]];
                    out_hw[[out_p.y as usize, out_p.x as usize]] = normalized_pixel;
                }

                let line_img_width = resized_line_width(line_rect.width(), line_rect.height());
                let resized_line_img = resize(
                    out_img.view(),
                    ResizeTarget::Sizes(&tensor!([1, 1, rec_img_height as i32, line_img_width])),
                    ResizeMode::Linear,
                    CoordTransformMode::default(),
                    NearestMode::default(),
                )
                .unwrap();

                // Copy resized line image to line recognition model input.
                line_rec_input
                    .slice_mut((group_line_index, 0, .., 0..(line_img_width as usize)))
                    .copy_from(&resized_line_img.squeezed());
            }

            // Perform text recognition on the line batch.
            let rec_output = model
                .run(
                    &[(rec_input_id, (&line_rec_input).into())],
                    &[rec_output_id],
                    None,
                )
                .unwrap();

            // Extract sequence as [seq, batch, class]
            let mut rec_sequence = rec_output[0].as_float_ref().unwrap().to_tensor();

            // Transpose to [batch, seq, class]
            rec_sequence.permute(&[1, 0, 2]);

            line_indices
                .iter()
                .copied()
                .enumerate()
                .map(move |(group_line_index, line_index)| {
                    let decoder = CtcDecoder::new();
                    let input_seq = rec_sequence.slice([group_line_index]);
                    let decode_result = decoder.decode_greedy(input_seq.clone());
                    (line_index, decode_result)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let line_texts = (0..lines.len())
        .filter_map(|idx| line_rec_results.get(&idx))
        .map(|result| result.to_string(DEFAULT_ALPHABET))
        .collect();

    Ok(line_texts)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut debug = false;
    let args: Vec<_> = std::env::args()
        .filter(|cmd| match cmd.as_ref() {
            "--debug" => {
                debug = true;
                false
            }
            _ => true,
        })
        .collect();

    if args.len() <= 1 {
        println!(
            "Usage: {} <detection_model> <rec_model> <image>",
            args.get(0).map(|s| s.as_str()).unwrap_or("")
        );
        // Exit with non-zero status, but also don't show an error.
        std::process::exit(1);
    }

    let detection_model_name = args.get(1).ok_or("detection model name not specified")?;
    let recognition_model_name = args.get(2).ok_or("recognition model name not specified")?;
    let image_path = args.get(3).ok_or("image path not specified")?;

    let detection_model_bytes = fs::read(detection_model_name)?;
    let detection_model = Model::load(&detection_model_bytes)?;

    let recognition_model_bytes = fs::read(recognition_model_name)?;
    let recognition_model = Model::load(&recognition_model_bytes)?;

    // Read image into CHW tensor.
    let color_img = read_image(image_path).expect("failed to read input image");

    let normalize_pixel = |pixel| pixel + ZERO_VALUE;

    // Convert color CHW tensor to fixed-size greyscale NCHW input expected by model.
    let grey_img = greyscale_image(color_img.view(), normalize_pixel);

    // Find text components (words) in the input image.
    let word_rects = detect_words(grey_img.view(), &detection_model, debug)?;

    // Perform layout analysis to group words into lines, in reading order.
    let [_, img_height, img_width] = grey_img.dims();
    let page_rect = Rect::from_hw(img_height as i32, img_width as i32);
    let line_rects = find_text_lines(&word_rects, page_rect);

    // Perform recognition on the detected text lines.
    let line_texts = recognize_text_lines(
        grey_img.view(),
        &line_rects,
        page_rect,
        &recognition_model,
        debug,
    )?;
    for line in line_texts {
        println!("{}", line);
    }

    if debug {
        println!(
            "Found {} words, {} lines in image of size {}x{}",
            word_rects.len(),
            line_rects.len(),
            img_width,
            img_height
        );

        let mut annotated_img = color_img;

        for line in line_rects {
            let line_poly = Polygon::new(line_polygon(&line));
            draw_polygon(annotated_img.nd_slice_mut([0]), line_poly.vertices(), 0.9); // Red
            draw_polygon(annotated_img.nd_slice_mut([1]), line_poly.vertices(), 0.); // Green
            draw_polygon(annotated_img.nd_slice_mut([2]), line_poly.vertices(), 0.);
            // Blue
        }

        // Write out the annotated input image.
        write_image("ocr-detection-output.png", annotated_img.view())?;
    }

    Ok(())
}
