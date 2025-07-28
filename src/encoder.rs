use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Module, VarBuilder};
use crate::attention::MultiHeadAttention;
use crate::feed_forward::FeedForward;

pub struct EncoderLayer {
    self_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl EncoderLayer {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_rate: f32,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attention = MultiHeadAttention::new(d_model, num_heads, dropout_rate, vb.pp("self_attention"))?;
        let feed_forward = FeedForward::new(d_model, d_ff, dropout_rate, vb.pp("feed_forward"))?;
        let norm1 = layer_norm(d_model, layer_norm_eps, vb.pp("norm1"))?;
        let norm2 = layer_norm(d_model, layer_norm_eps, vb.pp("norm2"))?;

        Ok(Self {
            self_attention,
            feed_forward,
            norm1,
            norm2,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, training: bool) -> Result<Tensor> {
        // Self-attention with residual connection and layer norm
        let attn_output = self.self_attention.forward(x, x, x, mask, training)?;
        let x = self.norm1.forward(&x.add(&attn_output)?)?;

        // Feed-forward with residual connection and layer norm
        let ff_output = self.feed_forward.forward(&x, training)?;
        self.norm2.forward(&x.add(&ff_output)?)
    }
}

pub struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_rate: f32,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            layers.push(EncoderLayer::new(
                d_model,
                num_heads,
                d_ff,
                dropout_rate,
                layer_norm_eps,
                vb.pp(format!("layer_{}", i)),
            )?);
        }

        Ok(Self { layers })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, training: bool) -> Result<Tensor> {
        let mut output = x.clone();
        for layer in &self.layers {
            output = layer.forward(&output, mask, training)?;
        }
        Ok(output)
    }
}
