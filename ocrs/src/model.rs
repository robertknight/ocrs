use rten_tensor::{Tensor, TensorView};

use crate::errors::ModelError;

/// Interface for running an ML model.
pub trait Model {
    /// Return the expected input shape as a mix of fixed and dynamic-sized
    /// dimensions.
    fn input_shape(&self) -> Result<Vec<rten::Dimension>, ModelError>;

    /// Run the model and return inference outputs.
    fn run(
        &self,
        input: TensorView<f32>,
        opts: Option<rten::RunOptions>,
    ) -> Result<Tensor<f32>, ModelError>;
}

impl Model for rten::Model {
    fn input_shape(&self) -> Result<Vec<rten::Dimension>, ModelError> {
        let input_id = self
            .input_ids()
            .first()
            .copied()
            .ok_or(ModelError::NoInputs)?;
        let input_shape = self
            .node_info(input_id)
            .and_then(|info| info.shape())
            .ok_or(ModelError::NoInputShape)?;
        Ok(input_shape)
    }

    fn run(
        &self,
        input: TensorView<f32>,
        opts: Option<rten::RunOptions>,
    ) -> Result<Tensor<f32>, ModelError> {
        let output = self
            .run_one(input.into(), opts)
            .map_err(|err| ModelError::RunFailed(err.into()))?
            .try_into()
            .map_err(|err: rten::TryFromValueError| ModelError::WrongOutput(err.to_string()))?;
        Ok(output)
    }
}
