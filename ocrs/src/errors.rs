use std::error::Error;
use std::fmt;

/// The error type returned when running a machine learning model fails.
#[derive(Debug)]
pub enum ModelRunError {
    /// Model execution failed.
    RunFailed(Box<dyn Error + Send + Sync>),

    /// The model output had a different data type or shape than expected.
    WrongOutput(String),
}

impl fmt::Display for ModelRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            ModelRunError::RunFailed(err) => write!(f, "model run failed: {}", err),
            ModelRunError::WrongOutput(err) => {
                write!(f, "model output had unexpected type or shape: {}", err)
            }
        }
    }
}

impl Error for ModelRunError {}
