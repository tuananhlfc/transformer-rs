use candle_core::{Device, Result, Tensor};
use candle_nn::Optimizer;

use crate::model::Transformer;

pub fn run_epoch<O: Optimizer>(
    data_loader: impl IntoIterator<Item = (Tensor, Tensor)>,
    model: &Transformer,
    loss_fn: &dyn Fn(&Tensor, &Tensor) -> Result<Tensor>,
    optimizer: &mut O,
    device: &Device,
) -> Result<f64> {
    let mut total_loss = 0.0;
    let mut total_tokens = 0;

    for (batch, target) in data_loader {
        let batch = batch.to_device(device)?;
        let target = target.to_device(device)?;

        // Forward pass
        let output = model.forward(&batch, &target, None, true)?;

        // Compute loss
        let loss = loss_fn(&output, &target)?;

        optimizer.backward_step(&loss)?;

        // Update statistics
        total_loss += loss.to_scalar::<f32>()? as f64;
        total_tokens += batch.dims()[0];
    }

    Ok(total_loss / total_tokens as f64)
}
