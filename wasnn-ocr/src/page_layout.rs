use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::iter::zip;

use wasnn_imageproc::{
    bounding_rect, find_contours, min_area_rect, simplify_polygon, BoundingRect, Line, Point, Rect,
    RetrievalMode, RotatedRect, Vec2,
};
use wasnn_tensor::NdTensorView;

struct Partition {
    score: f32,
    boundary: Rect,
    obstacles: Vec<Rect>,
}

impl PartialEq for Partition {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Partition {}

impl Ord for Partition {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
    }
}

impl PartialOrd for Partition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Iterator over empty rectangles within a rectangular boundary that contains
/// a set of "obstacles". See [max_empty_rects].
///
/// The order in which rectangles are returned is determined by a scoring
/// function `S`.
pub struct MaxEmptyRects<S>
where
    S: Fn(Rect) -> f32,
{
    queue: BinaryHeap<Partition>,
    score: S,
    min_width: u32,
    min_height: u32,
}

impl<S> MaxEmptyRects<S>
where
    S: Fn(Rect) -> f32,
{
    fn new(obstacles: &[Rect], boundary: Rect, score: S, min_width: u32, min_height: u32) -> Self {
        let mut queue = BinaryHeap::new();

        // Sort obstacles by X then Y coord. This means that when we choose a pivot
        // from any sub-sequence of `obstacles` we'll also be biased towards picking
        // a central obstacle.
        let mut obstacles = obstacles.to_vec();
        obstacles.sort_by_key(|o| {
            let c = o.center();
            (c.x, c.y)
        });

        if !boundary.is_empty() {
            queue.push(Partition {
                score: score(boundary),
                boundary,
                obstacles: obstacles.to_vec(),
            });
        }

        MaxEmptyRects {
            queue,
            score,
            min_width,
            min_height,
        }
    }
}

impl<S> Iterator for MaxEmptyRects<S>
where
    S: Fn(Rect) -> f32,
{
    type Item = Rect;

    fn next(&mut self) -> Option<Rect> {
        // Assuming the obstacle list is sorted, eg. by X coordinate, choose
        // a pivot that is in the middle.
        let choose_pivot = |r: &[Rect]| r[r.len() / 2];

        while let Some(part) = self.queue.pop() {
            let Partition {
                score: _,
                boundary: b,
                obstacles,
            } = part;

            if obstacles.is_empty() {
                return Some(b);
            }

            let pivot = choose_pivot(&obstacles);
            let right_rect = Rect::from_tlbr(b.top(), pivot.right(), b.bottom(), b.right());
            let left_rect = Rect::from_tlbr(b.top(), b.left(), b.bottom(), pivot.left());
            let top_rect = Rect::from_tlbr(b.top(), b.left(), pivot.top(), b.right());
            let bottom_rect = Rect::from_tlbr(pivot.bottom(), b.left(), b.bottom(), b.right());

            let sub_rects = [top_rect, left_rect, bottom_rect, right_rect];

            for sr in sub_rects {
                if (sr.width().max(0) as u32) < self.min_width
                    || (sr.height().max(0) as u32) < self.min_height
                    || sr.is_empty()
                {
                    continue;
                }

                let sr_obstacles: Vec<_> = obstacles
                    .iter()
                    .filter(|o| o.intersects(sr))
                    .copied()
                    .collect();

                // There should always be fewer obstacles in `sr` since it should
                // not intersect the pivot.
                assert!(sr_obstacles.len() < obstacles.len());

                self.queue.push(Partition {
                    score: (self.score)(sr),
                    obstacles: sr_obstacles,
                    boundary: sr,
                });
            }
        }

        None
    }
}

/// Return an iterator over empty rects in `boundary`, ordered by decreasing
/// value of the `score` function.
///
/// The `score` function must have the property that for any rectangle R and
/// sub-rectangle S that is contained within R, `score(S) <= score(R)`. A
/// typical score function would be the area of the rect, but other functions
/// can be used to favor different aspect ratios.
///
/// `min_width` and `min_height` specify thresholds on the size of rectangles
/// yielded by the iterator.
///
/// The implementation is based on algorithms from [^1].
///
/// [^1]: Breuel, Thomas M. “Two Geometric Algorithms for Layout Analysis.”
///       International Workshop on Document Analysis Systems (2002).
pub fn max_empty_rects<S>(
    obstacles: &[Rect],
    boundary: Rect,
    score: S,
    min_width: u32,
    min_height: u32,
) -> MaxEmptyRects<S>
where
    S: Fn(Rect) -> f32,
{
    MaxEmptyRects::new(obstacles, boundary, score, min_width, min_height)
}

/// Iterator adapter which filters rectangles that overlap rectangles already
/// returned by more than a certain amount.
pub trait FilterOverlapping {
    type Output: Iterator<Item = Rect>;

    /// Create an iterator which filters out rectangles that overlap those
    /// already returned by more than `factor`.
    ///
    /// `factor` is the minimum Intersection-over-Union ratio or Jaccard index [^1].
    /// See also [Rect::iou].
    ///
    /// [^1]: <https://en.wikipedia.org/wiki/Jaccard_index>
    fn filter_overlapping(self, factor: f32) -> Self::Output;
}

/// Implementation of [FilterOverlapping].
pub struct FilterRectIter<I: Iterator<Item = Rect>> {
    source: I,

    /// Rectangles already found.
    found: Vec<Rect>,

    /// Intersection-over-Union threshold.
    overlap_threshold: f32,
}

impl<I: Iterator<Item = Rect>> FilterRectIter<I> {
    fn new(source: I, overlap_threshold: f32) -> FilterRectIter<I> {
        FilterRectIter {
            source,
            found: Vec::new(),
            overlap_threshold,
        }
    }
}

impl<I: Iterator<Item = Rect>> Iterator for FilterRectIter<I> {
    type Item = Rect;

    fn next(&mut self) -> Option<Rect> {
        while let Some(r) = self.source.next() {
            if self
                .found
                .iter()
                .any(|f| f.iou(r) >= self.overlap_threshold)
            {
                continue;
            }
            self.found.push(r);
            return Some(r);
        }
        None
    }
}

impl<I: Iterator<Item = Rect>> FilterOverlapping for I {
    type Output = FilterRectIter<I>;

    fn filter_overlapping(self, factor: f32) -> Self::Output {
        FilterRectIter::new(self, factor)
    }
}

fn vec_to_point(v: Vec2) -> Point {
    Point::from_yx(v.y as i32, v.x as i32)
}

fn rects_separated_by_line(a: &RotatedRect, b: &RotatedRect, l: Line) -> bool {
    let a_to_b = Line::from_endpoints(vec_to_point(a.center()), vec_to_point(b.center()));
    a_to_b.intersects(l)
}

fn rightmost_edge(r: &RotatedRect) -> Line {
    let mut corners = r.corners();
    corners.sort_by_key(|p| p.x);
    Line::from_endpoints(corners[2], corners[3])
}

fn leftmost_edge(r: &RotatedRect) -> Line {
    let mut corners = r.corners();
    corners.sort_by_key(|p| p.x);
    Line::from_endpoints(corners[0], corners[1])
}

/// Group rects into lines. Each line is a chain of oriented rects ordered
/// left-to-right.
///
/// `separators` is a list of line segments that prevent the formation of
/// lines which cross them. They can be used to specify column boundaries
/// for example.
pub fn group_into_lines(rects: &[RotatedRect], separators: &[Line]) -> Vec<Vec<RotatedRect>> {
    let mut sorted_rects: Vec<_> = rects.to_vec();
    sorted_rects.sort_by_key(|r| r.bounding_rect().left());

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
                        && edge.center().x - last_edge.center().x >= -max_h_overlap
                        && last_edge.vertical_overlap(edge) >= overlap_threshold
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

/// Find the minimum-area oriented rectangles containing each connected
/// component in the binary mask `mask`.
pub fn find_connected_component_rects(
    mask: NdTensorView<i32, 2>,
    expand_dist: f32,
) -> Vec<RotatedRect> {
    // Threshold for the minimum area of returned rectangles.
    //
    // This can be used to filter out rects created by small false positives in
    // the mask, at the risk of filtering out true positives. The more accurate
    // the model producing the mask is, the less this is needed.
    let min_area_threshold = 100.;

    find_contours(mask, RetrievalMode::External)
        .iter()
        .filter_map(|poly| {
            let simplified = simplify_polygon(poly, 2. /* epsilon */);

            min_area_rect(&simplified).map(|mut rect| {
                rect.resize(
                    rect.width() + 2. * expand_dist,
                    rect.height() + 2. * expand_dist,
                );
                rect
            })
        })
        .filter(|r| r.area() >= min_area_threshold)
        .collect()
}

/// A text line is a sequence of RotatedRects for words, organized from left to
/// right.
type TextLine = Vec<RotatedRect>;

type TextParagraph = Vec<TextLine>;

/// Find separators between columns.
pub fn find_column_separators(words: &[RotatedRect]) -> Vec<Rect> {
    let Some(page_rect) = bounding_rect(words.iter()) else {
        return Vec::new();
    };

    // Estimate spacing statistics
    let mut lines = group_into_lines(words, &[]);
    lines.sort_by_key(|l| l.first().unwrap().bounding_rect().top());

    let mut all_word_spacings = Vec::new();
    for line in lines.iter() {
        if line.len() > 1 {
            let mut spacings: Vec<_> = zip(line.iter(), line.iter().skip(1))
                .map(|(cur, next)| next.bounding_rect().left() - cur.bounding_rect().right())
                .collect();
            spacings.sort();
            all_word_spacings.extend_from_slice(&spacings);
        }
    }
    all_word_spacings.sort();

    let median_word_spacing = all_word_spacings
        .get(all_word_spacings.len() / 2)
        .copied()
        .unwrap_or(10);
    let median_height = words
        .get(words.len() / 2)
        .map(|r| r.height())
        .unwrap_or(10.)
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
    let object_bboxes: Vec<_> = words.iter().map(|r| r.bounding_rect()).collect();
    let min_width = (median_word_spacing * 3) / 2;
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
    let separator_lines: Vec<_> = find_column_separators(words)
        .iter()
        .map(|r| {
            let center = r.center();
            Line::from_endpoints(
                Point::from_yx(r.top(), center.x),
                Point::from_yx(r.bottom(), center.x),
            )
        })
        .collect();

    let mut lines = group_into_lines(words, &separator_lines);

    // Approximate a text line by the 1D line from the center of the left
    // edge of the first word, to the center of the right edge of the last word.
    let midpoint_line = |words: &[RotatedRect]| -> Line {
        assert!(!words.is_empty());
        Line::from_endpoints(
            words.first().unwrap().bounding_rect().left_edge().center(),
            words.last().unwrap().bounding_rect().right_edge().center(),
        )
    };

    // Sort lines by vertical position.
    lines.sort_by_key(|words| midpoint_line(words).center().y);

    let is_separated_by =
        |line_a: &[RotatedRect], line_b: &[RotatedRect], separators: &[Line]| -> bool {
            let mid_a = midpoint_line(line_a);
            let mid_b = midpoint_line(line_b);
            let a_to_b = Line::from_endpoints(mid_a.center(), mid_b.center());
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

        let mut index = 0;
        while index < lines.len() {
            if !is_separated_by(&seed, &lines[index], &separator_lines) {
                para.push(lines.remove(index));
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

/// Normalize a line so that it's endpoints are sorted from top to bottom.
fn downwards_line(l: Line) -> Line {
    if l.start.y <= l.end.y {
        l
    } else {
        Line::from_endpoints(l.end, l.start)
    }
}

/// Return a polygon which contains all the rects in `words`.
///
/// `words` is assumed to be a series of disjoint rectangles ordered from left
/// to right. The returned points are arranged in clockwise order starting from
/// the top-left point.
///
/// There are several ways to compute a polygon for a line. The simplest is
/// to use [min_area_rect] on the union of the line's points. However the result
/// will not tightly fit curved lines. This function returns a polygon which
/// closely follows the edges of individual words.
pub fn line_polygon(words: &[RotatedRect]) -> Vec<Point> {
    let mut polygon = Vec::new();

    // Add points from top edges, in left-to-right order.
    for word_rect in words.iter() {
        let (left, right) = (
            downwards_line(leftmost_edge(word_rect)),
            downwards_line(rightmost_edge(word_rect)),
        );
        polygon.push(left.start);
        polygon.push(right.start);
    }

    // Add points from bottom edges, in right-to-left order.
    for word_rect in words.iter().rev() {
        let (left, right) = (
            downwards_line(leftmost_edge(word_rect)),
            downwards_line(rightmost_edge(word_rect)),
        );
        polygon.push(right.end);
        polygon.push(left.end);
    }

    polygon
}

#[cfg(test)]
mod tests {
    use wasnn_imageproc::{fill_rect, BoundingRect, Point, Polygon, Rect, RotatedRect, Vec2};
    use wasnn_tensor::NdTensor;

    use super::max_empty_rects;
    use crate::page_layout::{find_connected_component_rects, find_text_lines, line_polygon};

    /// Generate a grid of uniformly sized and spaced rects.
    ///
    /// `grid_shape` is a (rows, columns) tuple. `rect_size` and `gap_size` are
    /// (height, width) tuples.
    fn gen_rect_grid(
        top_left: Point,
        grid_shape: (i32, i32),
        rect_size: (i32, i32),
        gap_size: (i32, i32),
    ) -> Vec<Rect> {
        let mut rects = Vec::new();

        let (rows, cols) = grid_shape;
        let (rect_h, rect_w) = rect_size;
        let (gap_h, gap_w) = gap_size;

        for r in 0..rows {
            for c in 0..cols {
                let top = top_left.y + r * (rect_h + gap_h);
                let left = top_left.x + c * (rect_w + gap_w);
                rects.push(Rect::from_tlbr(top, left, top + rect_h, left + rect_w))
            }
        }

        rects
    }

    /// Return the union of `rects` or `None` if rects is empty.
    fn union_rects(rects: &[Rect]) -> Option<Rect> {
        rects
            .iter()
            .fold(None, |union, r| union.map(|u| u.union(*r)).or(Some(*r)))
    }

    #[test]
    fn test_max_empty_rects() {
        // Create a collection of obstacles that are laid out roughly like
        // words in a two-column document.
        let page = Rect::from_tlbr(0, 0, 80, 90);

        let left_col = gen_rect_grid(
            Point::from_yx(0, 0),
            /* grid_shape */ (10, 5),
            /* rect_size */ (5, 5),
            /* gap_size */ (3, 2),
        );
        let left_col_boundary = union_rects(&left_col).unwrap();
        assert!(page.contains(left_col_boundary));

        let right_col = gen_rect_grid(
            Point::from_yx(0, left_col_boundary.right() + 20),
            /* grid_shape */ (10, 5),
            /* rect_size */ (5, 5),
            /* gap_size */ (3, 2),
        );

        let right_col_boundary = union_rects(&right_col).unwrap();
        assert!(page.contains(right_col_boundary));

        let mut all_cols = left_col.clone();
        all_cols.extend_from_slice(&right_col);

        let max_area_rect = max_empty_rects(&all_cols, page, |r| r.area() as f32, 0, 0).next();

        assert_eq!(
            max_area_rect,
            Some(Rect::from_tlbr(
                page.top(),
                left_col_boundary.right(),
                page.bottom(),
                right_col_boundary.left()
            ))
        );
    }

    #[test]
    fn test_max_empty_rects_if_none() {
        // Case with no empty space within the boundary
        let boundary = Rect::from_tlbr(0, 0, 5, 5);
        assert_eq!(
            max_empty_rects(&[boundary], boundary, |r| r.area() as f32, 0, 0).next(),
            None
        );

        // Case where boundary is empty
        let boundary = Rect::from_hw(0, 0);
        assert_eq!(
            max_empty_rects(&[], boundary, |r| r.area() as f32, 0, 0).next(),
            None
        );
    }

    #[test]
    fn test_find_connected_component_rects() {
        let mut mask = NdTensor::zeros([400, 400]);
        let (grid_h, grid_w) = (5, 5);
        let (rect_h, rect_w) = (10, 50);
        let rects = gen_rect_grid(
            Point::from_yx(10, 10),
            (grid_h, grid_w), /* grid_shape */
            (rect_h, rect_w), /* rect_size */
            (10, 5),          /* gap_size */
        );
        for r in rects.iter() {
            // Expand `r` because `fill_rect` does not set points along the
            // right/bottom boundary.
            let expanded = r.adjust_tlbr(0, 0, 1, 1);
            fill_rect(mask.view_mut(), expanded, 1);
        }

        let components = find_connected_component_rects(mask.view(), 0.);
        assert_eq!(components.len() as i32, grid_h * grid_w);
        for c in components.iter() {
            let mut shape = [c.height().round() as i32, c.width().round() as i32];
            shape.sort();

            // We sort the dimensions before comparison here to be invariant to
            // different rotations of the connected component that cover the
            // same pixels.
            let mut expected_shape = [rect_h, rect_w];
            expected_shape.sort();

            assert_eq!(shape, expected_shape);
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
            .map(RotatedRect::from_rect)
            .collect();

        let rng = fastrand::Rng::with_seed(1234);
        rng.shuffle(&mut words);
        let lines = find_text_lines(&words);

        assert_eq!(lines.len() as i32, col_rows * 2);
        for line in lines {
            assert_eq!(line.len() as i32, col_words);

            let bounding_rect: Option<Rect> = line.iter().fold(None, |br, r| match br {
                Some(br) => Some(br.union(r.bounding_rect())),
                None => Some(r.bounding_rect()),
            });
            let (line_height, line_width) = bounding_rect
                .map(|br| (br.height(), br.width()))
                .unwrap_or((0, 0));

            // FIXME - The actual width/heights vary by one pixel and hence not
            // all match the expected size. Investigate why this happens.
            assert!((line_height - word_h).abs() <= 1);
            let expected_width = col_words * (word_w + word_gap) - word_gap;
            assert!((line_width - expected_width).abs() <= 1);
        }
    }

    #[test]
    fn test_line_polygon() {
        let words: Vec<RotatedRect> = (0..5)
            .map(|i| {
                let center = Vec2::from_yx(10., i as f32 * 20.);
                let width = 10.;
                let height = 5.;

                // Vary the orientation of words. The output of `line_polygon`
                // should be invariant to different orientations of a RotatedRect
                // that cover the same pixels.
                let up = if i % 2 == 0 {
                    Vec2::from_yx(-1., 0.)
                } else {
                    Vec2::from_yx(1., 0.)
                };
                RotatedRect::new(center, up, width, height)
            })
            .collect();
        let poly = Polygon::new(line_polygon(&words));

        assert!(poly.is_simple());
        for word in words {
            assert!(poly.contains_pixel(word.bounding_rect().center()));
        }
    }
}
