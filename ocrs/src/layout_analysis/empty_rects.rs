use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rten_imageproc::Rect;

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
                obstacles: obstacles.clone(),
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

#[cfg(test)]
mod tests {
    use rten_imageproc::{Point, Rect};

    use super::max_empty_rects;
    use crate::test_util::{gen_rect_grid, union_rects};

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
}
