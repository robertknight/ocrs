use rten_imageproc::{Point, Rect};

/// Generate a grid of uniformly sized and spaced rects.
///
/// `grid_shape` is a (rows, columns) tuple. `rect_size` and `gap_size` are
/// (height, width) tuples.
pub fn gen_rect_grid(
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
