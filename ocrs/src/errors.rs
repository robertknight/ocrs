use std::error::Error;
use std::fmt;

type BoxError = Box<dyn Error + Send + Sync>;

/// Categories of error returned by [`OcrError::kind`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrErrorKind {
    /// An operation required a text detection model, but none was provided
    /// when the engine was created.
    DetectionModelNotLoaded,

    /// An operation required a text recognition model, but none was provided
    /// when the engine was created.
    RecognitionModelNotLoaded,

    /// A model's inputs or outputs did not match what the engine expects.
    IncompatibleModel,

    /// Running a model, or a preprocessing step for its inputs or outputs,
    /// failed.
    InferenceFailed,
}

impl fmt::Display for OcrErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let desc = match self {
            OcrErrorKind::DetectionModelNotLoaded => "detection model not loaded",
            OcrErrorKind::RecognitionModelNotLoaded => "recognition model not loaded",
            OcrErrorKind::IncompatibleModel => "incompatible model",
            OcrErrorKind::InferenceFailed => "inference failed",
        };
        f.write_str(desc)
    }
}

/// Error returned by [`OcrEngine`](crate::OcrEngine) operations.
///
/// Use [`kind`](OcrError::kind) to determine the category of error. The
/// underlying cause, if any, is available via [`Error::source`].
///
/// The default `Display` output reports the error type. The alternate form
/// (`{:#}`) includes the details of underlying causes.
#[derive(Debug)]
pub struct OcrError {
    kind: OcrErrorKind,
    message: Option<String>,
    source: Option<BoxError>,
}

impl OcrError {
    /// Return the category of this error.
    pub fn kind(&self) -> OcrErrorKind {
        self.kind
    }

    pub(crate) fn new(kind: OcrErrorKind) -> Self {
        OcrError {
            kind,
            message: None,
            source: None,
        }
    }

    pub(crate) fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    pub(crate) fn with_source(mut self, source: impl Into<BoxError>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Create an [`OcrErrorKind::InferenceFailed`] error with a given cause.
    pub(crate) fn inference_failed(source: impl Into<BoxError>) -> Self {
        Self::new(OcrErrorKind::InferenceFailed).with_source(source)
    }
}

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)?;
        if let Some(message) = &self.message {
            write!(f, ": {}", message)?;
        }
        if f.alternate() {
            let mut source = self.source();
            while let Some(err) = source {
                write!(f, ": {}", err)?;
                source = err.source();
            }
        }
        Ok(())
    }
}

impl Error for OcrError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|err| err as &(dyn Error + 'static))
    }
}

/// Internal error type for failures when querying or running a [`Model`](crate::model::Model).
#[derive(Debug, thiserror::Error)]
pub(crate) enum ModelError {
    #[error("model has no inputs")]
    NoInputs,

    #[error("model does not specify expected input shape")]
    NoInputShape,

    #[error("model input shape is not supported")]
    UnsupportedInputShape,

    #[error("model run failed")]
    RunFailed(#[source] BoxError),

    #[error("model output had unexpected type or shape: {0}")]
    WrongOutput(String),
}

impl From<ModelError> for OcrError {
    fn from(err: ModelError) -> OcrError {
        match err {
            ModelError::RunFailed(source) => OcrError::inference_failed(source),
            ModelError::NoInputs
            | ModelError::NoInputShape
            | ModelError::UnsupportedInputShape
            | ModelError::WrongOutput(_) => {
                OcrError::new(OcrErrorKind::IncompatibleModel).with_message(err.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::{ModelError, OcrError, OcrErrorKind};

    #[derive(Debug, thiserror::Error)]
    #[error("inner cause")]
    struct InnerError;

    #[test]
    fn test_display() {
        let err = OcrError::new(OcrErrorKind::DetectionModelNotLoaded);
        assert_eq!(err.to_string(), "detection model not loaded");

        let err: OcrError = ModelError::NoInputs.into();
        assert_eq!(err.kind(), OcrErrorKind::IncompatibleModel);
        assert_eq!(err.to_string(), "incompatible model: model has no inputs");
        assert!(err.source().is_none());
    }

    #[test]
    fn test_source_chain() {
        let err: OcrError = ModelError::RunFailed(Box::new(InnerError)).into();
        assert_eq!(err.kind(), OcrErrorKind::InferenceFailed);
        assert_eq!(err.to_string(), "inference failed");
        assert_eq!(format!("{err:#}"), "inference failed: inner cause");
        assert_eq!(err.source().unwrap().to_string(), "inner cause");
    }
}
