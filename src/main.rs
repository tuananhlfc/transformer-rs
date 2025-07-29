use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use transformer_rs::model::{Transformer, TransformerConfig};

fn main() -> anyhow::Result<()> {
    // Set up the device
    let device = Device::Cpu;
    // Create a VarBuilder to initialize the model weights
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // 1. Define the Transformer configuration
    let model_config = TransformerConfig::default();

    // 3. Instantiate the Transformer model
    let model = Transformer::new(model_config, device.clone(), vb)?;

    // 4. Create dummy input data
    let src = Tensor::randn(0f32, 1f32, (10, 10), &device).unwrap();
    let tgt = Tensor::randn(0f32, 1f32, (10, 2), &device).unwrap();

    // 5. Pass the input data to the Transformer model
    let output = model.forward(&src, &tgt, None, true)?;

    // 6. Print the output
    println!("Output: {:?}", output);

    Ok(())
}
