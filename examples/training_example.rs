use candle_core::{Device, DType, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use transformer_rs::model::{TransformerConfig};
use transformer_rs::training::train::{run_epoch};
use transformer_rs::Transformer;

fn main() -> candle_core::Result<()> {
    // Set up device
    let device = Device::Cpu;
    print!("Using device: {:?}", device);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Define model configuration
    let model_config = TransformerConfig::default();
    let model = Transformer::new(model_config, device.clone(), vb)?;

    // Create dummy input data
    let data_iter = (0..100).map(|_| {
        let src = Tensor::randn(0f32, 1f32, (10, 10), &device).unwrap();
        let tgt = Tensor::randn(0f32, 1f32, (10, 2), &device).unwrap();
        (src, tgt)
    });

    // Setup AdamW optimizer
    let params = ParamsAdamW {
        lr: 0.001,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

    // Define a simple loss function
    let loss_compute = |pred: &Tensor, target: &Tensor| loss::mse(pred, target);

    // Run training loop
    for epoch in 1..=10 {
        let loss = run_epoch(data_iter.clone(), &model, &loss_compute, &mut optimizer, &device)?;
        println!("Epoch {}: Loss = {:.4}", epoch, loss);
    }

    Ok(())
}
