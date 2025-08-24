use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::utils::scaled_dot_product_attention;

pub struct MultiHeadAttention {
    d_model: usize,
    num_heads: usize,
    d_k: usize,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    dropout_rate: f32,
}

impl MultiHeadAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        dropout_rate: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(candle_core::Error::Msg(
                "d_model must be divisible by num_heads".to_string(),
            ));
        }

        let d_k = d_model / num_heads;

        let w_q = candle_nn::linear(d_model, d_model, vb.pp("w_q"))?;
        let w_k = candle_nn::linear(d_model, d_model, vb.pp("w_k"))?;
        let w_v = candle_nn::linear(d_model, d_model, vb.pp("w_v"))?;
        let w_o = candle_nn::linear(d_model, d_model, vb.pp("w_o"))?;

        Ok(Self {
            d_model,
            num_heads,
            d_k,
            w_q,
            w_k,
            w_v,
            w_o,
            dropout_rate,
        })
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        training: bool,
    ) -> Result<Tensor> {
        let batch_size = query.dims()[0];
        let seq_len = query.dims()[1];

        // Linear transformations and reshape to (batch_size, num_heads, seq_len, d_k)
        let q = self
            .w_q
            .forward(query)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        let k = self
            .w_k
            .forward(key)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        let v = self
            .w_v
            .forward(value)?
            .reshape((batch_size, seq_len, self.num_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply scaled dot-product attention
        let attention_output =
            scaled_dot_product_attention(&q, &k, &v, mask, self.dropout_rate, training)?;

        // Concatenate heads and put through final linear layer
        let concat_attention =
            attention_output
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch_size, seq_len, self.d_model))?;

        self.w_o.forward(&concat_attention)
    }
}
