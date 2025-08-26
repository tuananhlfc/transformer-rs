use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub src_vocab_size: usize,
    pub tgt_vocab_size: usize,
    pub dropout: f32,
    pub layer_norm_eps: f64,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            num_heads: 8,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            d_ff: 2048,
            max_seq_len: 512,
            src_vocab_size: 1000,
            tgt_vocab_size: 1000,
            dropout: 0.1,
            layer_norm_eps: 1e-6,
        }
    }
}
