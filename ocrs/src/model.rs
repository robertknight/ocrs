use anyhow::anyhow;

use rten_tensor::{Tensor, TensorView};

/// Interface for running an ML model.
pub trait Model {
    /// Return the expected input shape as a mix of fixed and dynamic-sized
    /// dimensions.
    fn input_shape(&self) -> anyhow::Result<Vec<rten::Dimension>>;

    /// Run the model and return inference outputs.
    fn run(
        &self,
        input: TensorView<f32>,
        opts: Option<rten::RunOptions>,
    ) -> anyhow::Result<Tensor<f32>>;
}

impl Model for rten::Model {
    fn input_shape(&self) -> anyhow::Result<Vec<rten::Dimension>> {
        let input_id = self
            .input_ids()
            .first()
            .copied()
            .ok_or(anyhow!("model has no inputs"))?;
        let input_shape = self
            .node_info(input_id)
            .and_then(|info| info.shape())
            .ok_or(anyhow!("model does not specify expected input shape"))?;
        Ok(input_shape)
    }

    fn run(
        &self,
        input: TensorView<f32>,
        opts: Option<rten::RunOptions>,
    ) -> anyhow::Result<Tensor<f32>> {
        let output = self.run_one(input.into(), opts)?.try_into()?;
        Ok(output)
    }
}
