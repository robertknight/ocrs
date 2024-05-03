use rten_imageproc::{bounding_rect, BoundingRect, Line, LineF, Point, Rect, RotatedRect};

use crate::geom_util::{leftmost_edge, rightmost_edge};

mod empty_rects;
use empty_rects::{max_empty_rects, FilterOverlapping};

fn rects_separated_by_line(a: &RotatedRect, b: &RotatedRect, l: LineF) -> bool {
    let a_to_b = LineF::from_endpoints(a.center(), b.center());
    a_to_b.intersects(l)
}

/// Group rects into lines. Each line is a chain of oriented rects ordered
/// left-to-right, which may overlap.
///
/// `separators` is a list of line segments that prevent the formation of
/// lines which cross them. They can be used to specify column boundaries
/// for example.
pub fn group_into_lines(rects: &[RotatedRect], separators: &[LineF]) -> Vec<Vec<RotatedRect>> {
    let mut sorted_rects: Vec<_> = rects.to_vec();
    sorted_rects.sort_by_key(|r| r.bounding_rect().left() as i32);

    let mut lines: Vec<Vec<_>> = Vec::new();

    // Minimum amount by which two words must overlap vertically to be
    // considered part of the same line.
    let overlap_threshold = 5;

    // Maximum amount by which a candidate word to extend a line from
    // left-to-right may horizontally overlap the current last word in the line.
    //
    // This is necessary when the code that produces the input rects can create
    // overlapping rects. `find_connected_component_rects` pads the rects it
    // produces for example.
    let max_h_overlap = 5;

    while !sorted_rects.is_empty() {
        let mut line = Vec::new();
        line.push(sorted_rects.remove(0));

        // Find the best candidate to extend the current line by one word to the
        // right, and keep going as long as we can find such a candidate.
        loop {
            let last = line.last().unwrap();
            let last_edge = rightmost_edge(last);

            if let Some((i, next_item)) = sorted_rects
                .iter()
                .enumerate()
                .filter(|(_, r)| {
                    let edge = leftmost_edge(r);
                    r.center().x > last.center().x
                        && edge.center().x - last_edge.center().x >= -max_h_overlap as f32
                        && last_edge.vertical_overlap(edge) >= overlap_threshold as f32
                        && !separators
                            .iter()
                            .any(|&s| rects_separated_by_line(last, r, s))
                })
                .min_by_key(|(_, r)| r.center().x as i32)
            {
                line.push(*next_item);
                sorted_rects.remove(i);
            } else {
                break;
            }
        }
        lines.push(line);
    }

    lines
}

/// A text line is a sequence of RotatedRects for words, organized from left to
/// right.
type TextLine = Vec<RotatedRect>;

type TextParagraph = Vec<TextLine>;

/// Find separators between text blocks.
///
/// This includes separators between columns, as well as between sections (eg.
/// headings and article contents).
pub fn find_block_separators(words: &[RotatedRect]) -> Vec<Rect> {
    let Some(page_rect) = bounding_rect(words.iter()).map(|br| br.integral_bounding_rect()) else {
        return Vec::new();
    };

    // Estimate spacing statistics
    let mut lines = group_into_lines(words, &[]);
    lines.sort_by_key(|l| l.first().unwrap().bounding_rect().top().round() as i32);

    let mut all_word_spacings = Vec::new();
    for line in lines {
        if line.len() > 1 {
            // `group_into_lines` sorts words in a line from left to right,
            // but they can overlap.
            let mut spacings: Vec<_> = line
                .iter()
                .zip(line.iter().skip(1))
                .map(|(cur, next)| {
                    (next.bounding_rect().left() - cur.bounding_rect().right())
                        .max(0.)
                        .round() as i32
                })
                .collect();
            spacings.sort_unstable();
            all_word_spacings.extend_from_slice(&spacings);
        }
    }
    all_word_spacings.sort_unstable();

    let median_word_spacing = all_word_spacings
        .get(all_word_spacings.len() / 2)
        .copied()
        .unwrap_or(10);
    let median_height = words
        .get(words.len() / 2)
        .map_or(10.0, |r| r.height())
        .round() as i32;

    // Scoring function for empty rectangles. Taken from Section 3.D in [1].
    // This favors tall rectangles.
    //
    // [1] F. Shafait, D. Keysers and T. Breuel, "Performance Evaluation and
    //     Benchmarking of Six-Page Segmentation Algorithms".
    //     10.1109/TPAMI.2007.70837.
    let score = |r: Rect| {
        let aspect_ratio = (r.height() as f32) / (r.width() as f32);
        let aspect_ratio_weight = match aspect_ratio.log2().abs() {
            r if r < 3. => 0.5,
            r if r < 5. => 1.5,
            r => r,
        };
        ((r.area() as f32) * aspect_ratio_weight).sqrt()
    };

    // Find separators between columns and articles.
    let object_bboxes: Vec<_> = words
        .iter()
        .map(|r| r.bounding_rect().integral_bounding_rect())
        .collect();
    let min_width = median_word_spacing * 3;
    let min_height = (3 * median_height.max(0)) as u32;

    max_empty_rects(
        &object_bboxes,
        page_rect,
        score,
        min_width.try_into().unwrap(),
        min_height,
    )
    .filter_overlapping(0.5)
    .take(80)
    .collect()
}

/// Group words into lines and sort them into reading order.
pub fn find_text_lines(words: &[RotatedRect]) -> Vec<Vec<RotatedRect>> {
    let separators = find_block_separators(words);
    let vertical_separators: Vec<_> = separators
        .iter()
        .map(|r| {
            let center = r.center();
            Line::from_endpoints(
                Point::from_yx(r.top(), center.x).to_f32(),
                Point::from_yx(r.bottom(), center.x).to_f32(),
            )
        })
        .collect();

    let horizontal_separators: Vec<_> = separators
        .iter()
        .map(|r| {
            let center = r.center();
            Line::from_endpoints(
                Point::from_yx(center.y, r.left()).to_f32(),
                Point::from_yx(center.y, r.right()).to_f32(),
            )
        })
        .collect();

    let mut lines = group_into_lines(words, &vertical_separators);

    // Approximate a text line by the 1D line from the center of the left
    // edge of the first word, to the center of the right edge of the last word.
    let midpoint_line = |words: &[RotatedRect]| -> LineF {
        assert!(!words.is_empty());
        Line::from_endpoints(
            words.first().unwrap().bounding_rect().left_edge().center(),
            words.last().unwrap().bounding_rect().right_edge().center(),
        )
    };

    // Sort lines by vertical position.
    lines.sort_by_key(|words| midpoint_line(words).center().y as i32);

    let is_separated_by = |line_a: LineF, line_b: LineF, separators: &[LineF]| -> bool {
        let a_to_b = Line::from_endpoints(line_a.center(), line_b.center());
        separators.iter().any(|sep| sep.intersects(a_to_b))
    };

    // Group lines into paragraphs. We repeatedly take the first un-assigned
    // line as the seed for a new paragraph, and then add to that para all
    // remaining un-assigned lines which are not separated from the seed.
    let mut paragraphs: Vec<TextParagraph> = Vec::new();
    while !lines.is_empty() {
        let seed = lines.remove(0);
        let mut para = Vec::new();
        para.push(seed.clone());

        let mut prev_line = midpoint_line(&seed);

        let mut index = 0;
        while index < lines.len() {
            let candidate_line = midpoint_line(&lines[index]);
            if prev_line.horizontal_overlap(candidate_line) > 0.
                && !is_separated_by(prev_line, candidate_line, &horizontal_separators)
            {
                para.push(lines.remove(index));
                prev_line = candidate_line;
            } else {
                index += 1;
            }
        }
        paragraphs.push(para);
    }

    // Flatten paragraphs into a list of lines.
    paragraphs
        .into_iter()
        .flat_map(|para| para.into_iter())
        .collect()
}

#[cfg(test)]
mod tests {
    use rten_imageproc::{BoundingRect, Point, Rect, RectF, RotatedRect};

    use super::{find_block_separators, find_text_lines};
    use crate::test_util::{gen_rect_grid, union_rects};

    #[test]
    fn test_find_block_separators() {
        struct Case {
            lines: i32,
            words: i32,
            word_h: i32,
            word_w: i32,
            line_gap: i32,
            word_gap: i32,
            expected_separators: usize,
        }

        let cases = [
            // Lines with overlapping words (negative `word_gap`).
            Case {
                lines: 2,
                words: 2,
                word_h: 10,
                word_w: 20,
                line_gap: 50,
                word_gap: -5,
                expected_separators: 2,
            },
        ];

        for Case {
            lines,
            words,
            word_h,
            word_w,
            line_gap,
            word_gap,
            expected_separators,
        } in cases
        {
            let words: Vec<RotatedRect> = gen_rect_grid(
                Point::from_yx(0, 0),
                (lines, words),
                (word_h, word_w),
                (line_gap, word_gap),
            )
            .into_iter()
            .map(|rect| RotatedRect::from_rect(rect.to_f32()))
            .collect();

            let separators = find_block_separators(&words);

            assert_eq!(separators.len(), expected_separators);
        }
    }

    #[test]
    fn test_find_text_lines() {
        // Create a collection of obstacles that are laid out roughly like
        // words in a two-column document.
        let page = Rect::from_tlbr(0, 0, 80, 90);
        let col_rows = 10;
        let col_words = 5;
        let (line_gap, word_gap) = (3, 2);
        let (word_h, word_w) = (5, 5);

        let left_col = gen_rect_grid(
            Point::from_yx(0, 0),
            /* grid_shape */ (col_rows, col_words),
            /* rect_size */ (word_h, word_w),
            /* gap_size */ (line_gap, word_gap),
        );
        let left_col_boundary = union_rects(&left_col).unwrap();
        assert!(page.contains(left_col_boundary));

        let right_col = gen_rect_grid(
            Point::from_yx(0, left_col_boundary.right() + 20),
            /* grid_shape */ (col_rows, col_words),
            /* rect_size */ (word_h, word_w),
            /* gap_size */ (line_gap, word_gap),
        );
        let right_col_boundary = union_rects(&right_col).unwrap();
        assert!(page.contains(right_col_boundary));

        let mut words: Vec<_> = left_col
            .iter()
            .chain(right_col.iter())
            .copied()
            .map(|r| RotatedRect::from_rect(r.to_f32()))
            .collect();

        let mut rng = fastrand::Rng::with_seed(1234);
        rng.shuffle(&mut words);
        let lines = find_text_lines(&words);

        assert_eq!(lines.len() as i32, col_rows * 2);
        for line in lines {
            assert_eq!(line.len() as i32, col_words);

            let bounding_rect: Option<RectF> = line.iter().fold(None, |br, r| match br {
                Some(br) => Some(br.union(r.bounding_rect())),
                None => Some(r.bounding_rect()),
            });
            let (line_height, line_width) = bounding_rect
                .map(|br| (br.height(), br.width()))
                .unwrap_or((0., 0.));

            // FIXME - The actual width/heights vary by one pixel and hence not
            // all match the expected size. Investigate why this happens.
            assert!((line_height - word_h as f32).abs() <= 1.);
            let expected_width = col_words * (word_w + word_gap) - word_gap;
            assert!((line_width - expected_width as f32).abs() <= 1.);
        }
    }
}
