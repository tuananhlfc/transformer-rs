use candle_core::{Result, Tensor, Device};
use candle_nn::{Dropout, LayerNorm, Module};
use super::attention::MultiHeadAttention;
use super::utils::FeedForward;

pub struct EncoderLayer {
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
}

impl EncoderLayer {
    pub fn new(
        d_model: usize,
        nhead: usize,
        d_ff: usize,
        dropout: f32,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(d_model, nhead, dropout, device)?,
            feed_forward: FeedForward::new(d_model, d_ff, dropout, device)?,
            norm1: LayerNorm::new(d_model, 1e-5, device)?,
            norm2: LayerNorm::new(d_model, 1e-5, device)?,
            dropout: Dropout::new(dropout),
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Self-attention + Add & Norm
        let attn_out = self.self_attn.forward(x, x, x, mask)?;
        let x = x.add(&self.dropout.forward(&attn_out, true)?)?;
        let x = self.norm1.forward(&x)?;

        // Feed-forward + Add & Norm
        let ff_out = self.feed_forward.forward(&x)?;
        let x = x.add(&self.dropout.forward(&ff_out, true)?)?;
        let x = self.norm2.forward(&x)?;

        Ok(x)
    }
}

pub struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub fn new(
        num_layers: usize,
        d_model: usize,
        nhead: usize,
        d_ff: usize,
        dropout: f32,
        device: &Device,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(EncoderLayer::new(d_model, nhead, d_ff, dropout, device)?);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x, mask)?;
        }
        Ok(x)
    }
}
