use std::fmt;
use std::fmt::Write;

use wasnn_imageproc::{bounding_rect, min_area_rect, Rect, RotatedRect};

/// A non-empty sequence of recognized characters ([TextChar]) that constitute a
/// logical unit of a document such as a word or line.
pub trait TextItem {
    /// Return the sequence of characters that make up this item.
    fn chars(&self) -> &[TextChar];

    /// Return the bounding rectangle of all characters in this item.
    fn bounding_rect(&self) -> Rect {
        bounding_rect(self.chars().iter().map(|c| &c.rect)).expect("expected valid rect")
    }

    /// Return the oriented bounding rectangle of all characters in this item.
    fn rotated_rect(&self) -> RotatedRect {
        let points: Vec<_> = self.chars().iter().flat_map(|c| c.rect.corners()).collect();
        min_area_rect(&points).expect("expected valid rect")
    }
}

fn fmt_text_item<TI: TextItem>(item: &TI, f: &mut fmt::Formatter) -> fmt::Result {
    for c in item.chars().iter().map(|c| c.char) {
        f.write_char(c)?;
    }
    Ok(())
}

impl fmt::Display for TextLine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt_text_item(self, f)
    }
}

/// Details of a single character that was recognized.
#[derive(Clone)]
pub struct TextChar {
    /// Character that was recognized.
    pub char: char,

    /// Approximate bounding rectangle of character in input image.
    pub rect: Rect,
}

/// Result of recognizing a line of text.
///
/// This includes the sequence of characters that were found and associated
/// metadata (eg. bounding boxes).
#[derive(Clone)]
pub struct TextLine {
    chars: Vec<TextChar>,
}

impl TextLine {
    pub(crate) fn new(chars: Vec<TextChar>) -> TextLine {
        assert!(!chars.is_empty(), "Text lines must not be empty");
        TextLine { chars }
    }

    /// Return an iterator over words in this line.
    pub fn words(&self) -> impl Iterator<Item = TextWord> {
        self.chars()
            .split(|c| c.char == ' ')
            .filter(|chars| !chars.is_empty())
            .map(TextWord::new)
    }
}

impl TextItem for TextLine {
    /// Return the bounding rects of each character in the line.
    fn chars(&self) -> &[TextChar] {
        &self.chars
    }
}

/// Subsequence of a [TextLine] that contains a sequence of non-space characters.
pub struct TextWord<'a> {
    chars: &'a [TextChar],
}

impl<'a> TextWord<'a> {
    fn new(chars: &'a [TextChar]) -> TextWord {
        assert!(!chars.is_empty(), "Text words must not be empty");
        TextWord { chars }
    }
}

impl<'a> TextItem for TextWord<'a> {
    fn chars(&self) -> &[TextChar] {
        self.chars
    }
}

impl<'a> fmt::Display for TextWord<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt_text_item(self, f)
    }
}
