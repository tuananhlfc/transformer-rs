use candle_core::{Result, Tensor, Device, DType};
use candle_nn::{Linear, Dropout, Module};

pub struct MultiHeadAttention {
    d_model: usize,
    nhead: usize,
    head_dim: usize,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn new(
        d_model: usize,
        nhead: usize,
        dropout: f32,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = d_model / nhead;
        Ok(Self {
            d_model,
            nhead,
            head_dim,
            w_q: Linear::new(d_model, d_model, device)?,
            w_k: Linear::new(d_model, d_model, device)?,
            w_v: Linear::new(d_model, d_model, device)?,
            w_o: Linear::new(d_model, d_model, device)?,
            dropout: Dropout::new(dropout),
        })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = (q.dims()[0], q.dims()[1], q.dims()[2]);
        let nhead = self.nhead;
        let head_dim = self.head_dim;

        // Linear projections
        let q = self.w_q.forward(q)?.reshape((batch_size, seq_len, nhead, head_dim))?.transpose(1, 2)?; // [B, nhead, S, head_dim]
        let k = self.w_k.forward(k)?.reshape((batch_size, seq_len, nhead, head_dim))?.transpose(1, 2)?;
        let v = self.w_v.forward(v)?.reshape((batch_size, seq_len, nhead, head_dim))?.transpose(1, 2)?;

        // Scaled dot-product attention
        let attn_scores = q.matmul(&k.transpose(-2, -1)?)? / (head_dim as f64).sqrt();
        let attn_scores = if let Some(mask) = mask {
            attn_scores.broadcast_add(mask)?
        } else {
            attn_scores
        };
        let attn_weights = attn_scores.softmax(-1)?;
        let attn_weights = self.dropout.forward(&attn_weights, true)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Concatenate heads
        let attn_output = attn_output.transpose(1, 2)?.reshape((batch_size, seq_len, nhead * head_dim))?;
        self.w_o.forward(&attn_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_multi_head_attention_shapes() {
        let device = Device::Cpu;
        let mha = MultiHeadAttention::new(8, 2, 0.1, &device).unwrap();
        let x = Tensor::zeros((2, 4, 8), &device).unwrap(); // [batch, seq, d_model]
        let y = mha.forward(&x, &x, &x, None).unwrap();
        assert_eq!(y.dims(), &[2, 4, 8]);
    }
}
