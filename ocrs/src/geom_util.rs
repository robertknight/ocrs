//! Shared utilities for working with shapes.

use rten_imageproc::{Coord, Line, LineF, RotatedRect};

/// Return the edge of a rotated rect consisting of the two right-most vertices.
pub fn rightmost_edge(r: &RotatedRect) -> LineF {
    let mut corners = r.corners();
    corners.sort_by(|a, b| a.x.total_cmp(&b.x));
    Line::from_endpoints(corners[2], corners[3])
}

/// Return the edge of a rotated rect consisting of the two left-most vertices.
pub fn leftmost_edge(r: &RotatedRect) -> LineF {
    let mut corners = r.corners();
    corners.sort_by(|a, b| a.x.total_cmp(&b.x));
    Line::from_endpoints(corners[0], corners[1])
}

/// Normalize a line so that it's endpoints are sorted from top to bottom.
pub fn downwards_line<T: Coord>(l: Line<T>) -> Line<T> {
    if l.start.y <= l.end.y {
        l
    } else {
        Line::from_endpoints(l.end, l.start)
    }
}
