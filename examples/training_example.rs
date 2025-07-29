use candle_core::Device;
use transformer_rs::model::{TransformerConfig};
use transformer_rs::training::train::{Trainer, TrainerConfig};

fn main() -> candle_core::Result<()> {
    // Set up device
    let device = Device::Cpu;

    // Define model configuration
    let model_config = TransformerConfig::default();

    // Define trainer configuration
    let trainer_config = TrainerConfig::default();

    // Create trainer
    let mut trainer = Trainer::new(model_config, trainer_config, &device)?;

    // Run training loop
    trainer.train()?;

    Ok(())
}
