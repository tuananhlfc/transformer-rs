use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use transformer_rs::model::{Transformer, TransformerConfig};

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;

    // 1. Create a Transformer configuration
    let config = TransformerConfig::default();

    // 2. Create a VarBuilder to initialize the model weights
    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);

    // 3. Instantiate the Transformer model
    let model = Transformer::new(config, device.clone(), vb)?;

    // 4. Create dummy input data
    let src = Tensor::ones((1, 10), candle_core::DType::U32, &device)?;
    let tgt = Tensor::ones((1, 12), candle_core::DType::U32, &device)?;

    // 5. Pass the input data to the Transformer model
    let output = model.forward(&src, &tgt, None, true)?;

    // 6. Print the output
    println!("Output: {:?}", output);

    Ok(())
}
