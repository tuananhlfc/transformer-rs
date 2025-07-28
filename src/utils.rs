use candle_core::{Device, Result, Tensor, D};

pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if i < j { f64::NEG_INFINITY } else { 0.0}))
        .collect();
    Tensor::from_vec(mask, (seq_len, seq_len), device)
}

pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    dropout_rate: f32,
    training: bool,
) -> Result<Tensor> {
    let d_k = query.dims()[query.dims().len() - 1] as f32;
    let scale = 1.0 / d_k.sqrt();
    
    // QK^T / sqrt(d_k)
    let scores = query.matmul(&key.transpose(D::Minus1, D::Minus2)?)?;
    let scaled_scores = scores.mul(&Tensor::new(scale, query.device())?)?;
    
    // Apply mask if provided
    let masked_scores = if let Some(mask) = mask {
        let neg_inf = Tensor::new(-1e9f32, scaled_scores.device())?;
        let mask_expanded = mask.broadcast_as(scaled_scores.shape())?;
        scaled_scores.where_cond(&mask_expanded, &neg_inf)?
    } else {
        scaled_scores
    };
    
    // Softmax
    let attention_weights = candle_nn::ops::softmax_last_dim(&masked_scores)?;
    
    // Apply dropout if training
    let attention_weights = if training && dropout_rate > 0.0 {
        let dropout = candle_nn::Dropout::new(dropout_rate);
        dropout.forward(&attention_weights, training)?
    } else {
        attention_weights
    };
    
    // Apply attention to values
    attention_weights.matmul(value)
}
