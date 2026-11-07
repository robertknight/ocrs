//! Minimal URL parsing.
//!
//! This supports only the subset of URL parsing needed by ocrs-cli. URLs are
//! required to be ASCII. No normalization or percent-decoding is performed.

use std::fmt;

/// Error returned when a URL cannot be parsed.
#[derive(Clone, Debug, PartialEq)]
pub enum ParseError {
    /// The URL contains non-ASCII characters.
    NonAscii,
    /// The URL is missing a scheme or the scheme is invalid.
    InvalidScheme,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::NonAscii => write!(f, "URL contains non-ASCII characters"),
            ParseError::InvalidScheme => write!(f, "URL has a missing or invalid scheme"),
        }
    }
}

impl std::error::Error for ParseError {}

/// A parsed absolute URL of the form
/// `scheme:[//authority]path[?query][#fragment]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Url<'a> {
    path: &'a str,
}

impl<'a> Url<'a> {
    /// Parse an absolute ASCII URL.
    pub fn parse(url: &'a str) -> Result<Url<'a>, ParseError> {
        let url = url.trim_matches(|c: char| c <= ' ');
        if !url.is_ascii() {
            return Err(ParseError::NonAscii);
        }

        let (scheme, rest) = url.split_once(':').ok_or(ParseError::InvalidScheme)?;
        let scheme_valid = scheme
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic())
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
        if !scheme_valid {
            return Err(ParseError::InvalidScheme);
        }

        // Strip query and fragment.
        let rest = rest.find(['?', '#']).map_or(rest, |end| &rest[..end]);

        // Strip authority.
        let path = match rest.strip_prefix("//") {
            Some(after_slashes) => after_slashes
                .find('/')
                .map_or("", |auth_end| &after_slashes[auth_end..]),
            None => rest,
        };

        Ok(Url { path })
    }

    /// Return the path component, excluding any query or fragment. The path
    /// is returned as-is, without percent-decoding.
    pub fn path(&self) -> &'a str {
        self.path
    }
}

#[cfg(test)]
mod tests {
    use super::{ParseError, Url};

    #[test]
    fn test_parse() {
        let url = Url::parse("https://models.com/dir/text-detection.onnx?v=1#frag").unwrap();
        assert_eq!(url.path(), "/dir/text-detection.onnx");

        let url = Url::parse("https://models.com").unwrap();
        assert_eq!(url.path(), "");

        let url = Url::parse("https://models.com?query").unwrap();
        assert_eq!(url.path(), "");

        let url = Url::parse("file:///tmp/model.rten").unwrap();
        assert_eq!(url.path(), "/tmp/model.rten");

        let url = Url::parse("data:foo/bar").unwrap();
        assert_eq!(url.path(), "foo/bar");

        let url = Url::parse(" \thttps://models.com/model.onnx\r\n").unwrap();
        assert_eq!(url.path(), "/model.onnx");
    }

    #[test]
    fn test_parse_invalid() {
        assert_eq!(Url::parse("no-scheme"), Err(ParseError::InvalidScheme));
        assert_eq!(Url::parse(":foo"), Err(ParseError::InvalidScheme));
        assert_eq!(Url::parse("1http://foo"), Err(ParseError::InvalidScheme));
        assert_eq!(Url::parse("ht tp://foo"), Err(ParseError::InvalidScheme));
        assert_eq!(
            Url::parse("https://example.com/caf\u{e9}"),
            Err(ParseError::NonAscii)
        );
    }
}
