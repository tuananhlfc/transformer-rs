use candle_core::{Result, Tensor};
use candle_nn::{Dropout, Linear, Module, VarBuilder};

pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, dropout_rate: f32, vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(d_model, d_ff, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(d_ff, d_model, vb.pp("linear2"))?;
        let dropout = Dropout::new(dropout_rate);

        Ok(Self {
            linear1,
            linear2,
            dropout,
        })
    }

    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.relu()?;
        let x = self.dropout.forward(&x, training)?;
        self.linear2.forward(&x)
    }
}
