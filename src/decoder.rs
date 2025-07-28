use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Module, VarBuilder};

use crate::{attention::MultiHeadAttention, feed_forward::FeedForward};


pub struct DecoderLayer {
    self_attention: MultiHeadAttention,
    cross_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

impl DecoderLayer {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_rate: f32,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attention = MultiHeadAttention::new(d_model, num_heads, dropout_rate, vb.pp("self_attention"))?;
        let cross_attention = MultiHeadAttention::new(d_model, num_heads, dropout_rate, vb.pp("cross_attention"))?;
        let feed_forward = FeedForward::new(d_model, d_ff, dropout_rate, vb.pp("feed_forward"))?;
        let norm1 = layer_norm(d_model, layer_norm_eps, vb.pp("norm1"))?;
        let norm2 = layer_norm(d_model, layer_norm_eps, vb.pp("norm2"))?;
        let norm3 = layer_norm(d_model, layer_norm_eps, vb.pp("norm3"))?;

        Ok(Self {
            self_attention,
            cross_attention,
            feed_forward,
            norm1,
            norm2,
            norm3,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        encoder_output: &Tensor,
        src_mask: Option<&Tensor>,
        tgt_mask: Option<&Tensor>,
        training: bool,
    ) -> Result<Tensor> {
        // Masked self-attention
        let self_attn_output = self.self_attention.forward(x, x, x, tgt_mask, training)?;
        let x = self.norm1.forward(&x.add(&self_attn_output)?)?;

        // Cross-attention
        let cross_attn_output = self.cross_attention.forward(
            &x, encoder_output, encoder_output, src_mask, training
        )?;
        let x = self.norm2.forward(&x.add(&cross_attn_output)?)?;

        // Feed-forward
        let ff_output = self.feed_forward.forward(&x, training)?;
        self.norm3.forward(&x.add(&ff_output)?)
    }
}

pub struct Decoder {
    layers: Vec<DecoderLayer>,
}

impl Decoder {
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
            layers.push(DecoderLayer::new(
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

    pub fn forward(
        &self,
        x: &Tensor,
        encoder_output: &Tensor,
        src_mask: Option<&Tensor>,
        tgt_mask: Option<&Tensor>,
        training: bool,
    ) -> Result<Tensor> {
        let mut output = x.clone();
        for layer in &self.layers {
            output = layer.forward(&output, encoder_output, src_mask, tgt_mask, training)?;
        }
        Ok(output)
    }
}
