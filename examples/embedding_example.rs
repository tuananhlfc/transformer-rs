use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use transformer_rs::model::embedding::Embeddings;

fn main() -> Result<()> {
    // Hyperparameters
    let vocab_size = 100;
    let d_model = 16;
    let max_seq_len = 32;
    let dropout_rate = 0.1;

    // Device
    let device = Device::Cpu;

    // Variable builder (empty for demonstration)
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Create Embeddings
    let embeddings = Embeddings::new(vocab_size, d_model, max_seq_len, dropout_rate, &device, vb)?;

    // Example input: batch of 2 sequences, each of length 5
    let input_ids = Tensor::from_slice(
        &[[1u32, 2, 3, 4, 5], [6, 7, 8, 9, 10]].concat(),
        (2, 5),
        &device,
    )?;

    // Forward pass
    let output = embeddings.forward(&input_ids)?;

    println!("Embeddings output shape: {:?}", output.dims());
    println!("Embeddings output: {:?}", output);

    // Print some actual values to verify they look reasonable
    println!("First few values from output:");
    let output_slice = output.narrow(0, 0, 1)?.narrow(1, 0, 2)?;
    println!("{:?}", output_slice.to_vec2::<f32>()?);

    Ok(())
}
