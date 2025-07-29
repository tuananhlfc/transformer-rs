use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};


use super::config::TransformerConfig;
use super::embedding::Embeddings;
use super::encoder::Encoder;
use super::decoder::Decoder;
use super::utils::causal_mask;

pub struct Transformer {
    config: TransformerConfig,
    src_embeddings: Embeddings,
    tgt_embeddings: Embeddings,
    encoder: Encoder,
    decoder: Decoder,
    generator: Linear,
    device: Device,
}

impl Transformer {
    pub fn new(
        config: TransformerConfig,
        device: Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let src_embeddings = Embeddings::new(
            config.src_vocab_size,
            config.d_model,
            config.max_seq_len,
            config.dropout,
            &device,
            vb.pp("src_embeddings"),
        )?;

        let tgt_embeddings = Embeddings::new(
            config.tgt_vocab_size,
            config.d_model,
            config.max_seq_len,
            config.dropout,
            &device,
            vb.pp("tgt_embeddings"),
        )?;

        let encoder = Encoder::new(
            config.num_encoder_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.dropout,
            config.layer_norm_eps,
            vb.pp("encoder"),
        )?;

        let decoder = Decoder::new(
            config.num_decoder_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.dropout,
            config.layer_norm_eps,
            vb.pp("decoder"),
        )?;

        let generator = candle_nn::linear(
            config.d_model,
            config.tgt_vocab_size,
            vb.pp("generator"),
        )?;

        Ok(Self {
            config,
            src_embeddings,
            tgt_embeddings,
            encoder,
            decoder,
            generator,
            device,
        })
    }

    pub fn forward(
        &self,
        src_input_ids: &Tensor,
        tgt_input_ids: &Tensor,
        src_mask: Option<&Tensor>,
        training: bool,
    ) -> Result<Tensor> {
        let tgt_seq_len = tgt_input_ids.dims()[1];
        let tgt_mask = causal_mask(tgt_seq_len, &self.device)?;

        // Encode source sequence
        let src_embeddings = self.src_embeddings.forward(src_input_ids)?;
        let encoder_output = self.encoder.forward(&src_embeddings, src_mask, training)?;

        // Decode target sequence
        let tgt_embeddings = self.tgt_embeddings.forward(tgt_input_ids)?;
        let decoder_output = self.decoder.forward(
            &tgt_embeddings,
            &encoder_output,
            src_mask,
            Some(&tgt_mask),
            training,
        )?;

        // Project to vocabulary
        self.generator.forward(&decoder_output)
    }

    pub fn encode(&self, src_input_ids: &Tensor, src_mask: Option<&Tensor>) -> Result<Tensor> {
        let src_embeddings = self.src_embeddings.forward(src_input_ids)?;
        self.encoder.forward(&src_embeddings, src_mask, false)
    }

    pub fn decode(
        &self,
        tgt_input_ids: &Tensor,
        encoder_output: &Tensor,
        src_mask: Option<&Tensor>,
        tgt_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let tgt_embeddings = self.tgt_embeddings.forward(tgt_input_ids)?;
        let decoder_output = self.decoder.forward(
            &tgt_embeddings,
            encoder_output,
            src_mask,
            tgt_mask,
            false,
        )?;
        self.generator.forward(&decoder_output)
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}
