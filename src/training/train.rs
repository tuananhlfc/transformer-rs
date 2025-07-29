use candle_core::{Result, Tensor, Device};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};

use crate::model::{Transformer, TransformerConfig};

pub struct TrainerConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
}

pub struct Trainer {
    model: Transformer,
    optimizer: AdamW,
    config: TrainerConfig,
    device: Device,
}

impl Trainer {
    pub fn new(
        model_config: TransformerConfig,
        trainer_config: TrainerConfig,
        device: &Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
        let model = Transformer::new(model_config.clone(), device.clone(), vb)?;
        let adamw_params = candle_nn::ParamsAdamW {
            lr: trainer_config.learning_rate,
            ..Default::default()
        };
        let optimizer = AdamW::new(varmap.all_vars(), adamw_params)?;
        Ok(Self {
            model,
            optimizer,
            config: trainer_config,
            device: device.clone(),
        })
    }

    pub fn train(&mut self) -> Result<()> {
        for epoch in 0..self.config.epochs {
            let (src, tgt) = self.dummy_data(self.config.batch_size, 10, 12)?;
            let output = self.model.forward(&src, &tgt, None, true)?;

            // TODO: Implement loss calculation
            let loss = output.sum_all()?;

            self.optimizer.backward_step(&loss)?;

            println!("Epoch: {}, Loss: {}", epoch, loss.to_scalar::<f32>()?);
        }
        Ok(())
    }

    fn dummy_data(&self, batch_size: usize, seq_len_src: usize, seq_len_tgt: usize) -> Result<(Tensor, Tensor)> {
        let src = Tensor::ones((batch_size, seq_len_src), candle_core::DType::U32, &self.device)?;
        let tgt = Tensor::ones((batch_size, seq_len_tgt), candle_core::DType::U32, &self.device)?;
        Ok((src, tgt))
    }
}
