use candle_core::{D, Device, Result, Tensor};

/// Manual softmax implementation that works with CUDA
/// Applies softmax along the last dimension
fn softmax_last_dim_manual(tensor: &Tensor) -> Result<Tensor> {
    let last_dim = tensor.dims().len() - 1;

    // Subtract max for numerical stability
    let max_vals = tensor.max_keepdim(last_dim)?;
    let shifted = tensor.broadcast_sub(&max_vals)?;

    // Compute exp
    let exp_vals = shifted.exp()?;

    // Compute sum along last dimension
    let sum_exp = exp_vals.sum_keepdim(last_dim)?;

    // Divide by sum to get probabilities
    exp_vals.broadcast_div(&sum_exp)
}

pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 }))
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
    let scale_tensor = Tensor::new(&[scale], scores.device())?;
    let scaled_scores = scores.broadcast_mul(&scale_tensor)?;

    // Apply mask if provided
    let masked_scores = if let Some(mask) = mask {
        // Add mask to scores instead of using where_cond
        let mask_expanded = mask.broadcast_as(scaled_scores.shape())?;
        scaled_scores.add(&mask_expanded)?
    } else {
        scaled_scores
    };

    // Softmax - manual implementation for CUDA compatibility
    let attention_weights: Tensor = softmax_last_dim_manual(&masked_scores)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Result};

    #[test]
    fn test_causal_mask_general() -> Result<()> {
        let device = Device::Cpu;
        let seq_len = 4;
        let mask = causal_mask(seq_len, &device)?;

        // Expected mas:
        // [[0.0, -inf, -inf, -inf],
        //  [0.0, 0.0, -inf, -inf],
        //  [0.0, 0.0, 0.0, -inf],
        //  [0.0, 0.0, 0.0, 0.0]]

        // Check the tensor's shape
        assert_eq!(mask.shape().dims(), &[seq_len, seq_len]);

        // Check tensor's content
        let expected = vec![
            vec![0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
            vec![0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
            vec![0.0, 0.0, 0.0, f32::NEG_INFINITY],
            vec![0.0, 0.0, 0.0, 0.0],
        ];
        let mask_data = mask.to_vec2::<f32>()?;
        assert_eq!(mask_data, expected, "Causal mask does not match expected values");
        Ok(())
    }
}
