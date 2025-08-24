use candle_core::{Device, DType, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use transformer_rs::model::{TransformerConfig};
use transformer_rs::training::train::{run_epoch};
use transformer_rs::Transformer;
use rand;

fn main() -> candle_core::Result<()> {
    // Set up device
    let device = Device::Cpu;
    print!("Using device: {:?}", device);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Define model configuration
    let model_config = TransformerConfig::default();
    let vocab_size = model_config.src_vocab_size; // Extract vocab_size before moving config
    let model = Transformer::new(model_config, device.clone(), vb)?;

    // Create dummy input data with proper integer token IDs
    // batch_size=2, seq_len=10, using vocabulary IDs from 1-999
    let batch_size = 2;
    let seq_len = 10;

    // Pre-generate training data to avoid device ownership issues
    let mut training_data = Vec::new();
    for _ in 0..100 {
        // Generate random token IDs for source sequence
        let src_data: Vec<u32> = (0..batch_size * seq_len)
            .map(|_| (rand::random::<u32>() % (vocab_size as u32 - 1)) + 1)
            .collect();
        let src = Tensor::from_vec(src_data, (batch_size, seq_len), &device)?;

        // Generate random token IDs for target sequence
        let tgt_data: Vec<u32> = (0..batch_size * seq_len)
            .map(|_| (rand::random::<u32>() % (vocab_size as u32 - 1)) + 1)
            .collect();
        let tgt = Tensor::from_vec(tgt_data, (batch_size, seq_len), &device)?;

        training_data.push((src, tgt));
    }

    // Setup AdamW optimizer
    let params = ParamsAdamW {
        lr: 0.001,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

    // Define a cross-entropy loss function
    // pred: [batch_size, seq_len, vocab_size], target: [batch_size, seq_len]
    let loss_compute = |pred: &Tensor, target: &Tensor| -> candle_core::Result<Tensor> {
        // Reshape pred from [batch, seq, vocab] to [batch*seq, vocab]
        let batch_size = pred.dims()[0];
        let seq_len = pred.dims()[1];
        let vocab_size = pred.dims()[2];

        let pred_flat = pred.reshape((batch_size * seq_len, vocab_size))?;
        let target_flat = target.reshape((batch_size * seq_len,))?;

        // Use negative log likelihood as a proxy for cross-entropy
        loss::nll(&pred_flat, &target_flat)
    };

    // Run training loop
    for epoch in 1..=10 {
        let loss = run_epoch(training_data.clone(), &model, &loss_compute, &mut optimizer, &device)?;
        println!("Epoch {}: Loss = {:.4}", epoch, loss);
    }

    Ok(())
}
