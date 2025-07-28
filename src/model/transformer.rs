use candle_core::{Result, Tensor, Device};
use candle_nn::{Linear, VarBuilder};
use crate::config::TransformerConfig;
use crate::model::{Embeddings, Encoder, Decoder};
use crate::utils::math::create_causal_mask;

pub struct Transformer {
    config: TransformerConfig,
    src_embeddings: Embeddings,
    tgt_embeddings: Embeddings,
    encoder: Encoder,
    decoder: Decoder,
    output_projection: Linear,
    device: Device,
}