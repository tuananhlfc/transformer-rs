use candle_core::{Device, Result, Tensor};
use candle_nn::{Dropout, Embedding, Module, VarBuilder};

pub struct PositionalEncoding {
    pe: Tensor,
    dropout: Dropout,
    d_model: usize,
    max_seq_len: usize,
}

pub struct Embeddings {
    word_embeddings: Embedding,
    positional_encoding: PositionalEncoding,
    d_model: usize,
}

impl PositionalEncoding {
    pub fn new(
        d_model: usize,
        max_seq_len: usize,
        dropout_rate: f32,
        device: &Device,
    ) -> Result<Self> {
        if d_model % 2 != 0 {
            return Err(candle_core::Error::Msg(
                "d_model must be even for positional encoding".to_string(),
            ));
        }

        let pe = Self::create_positional_encoding(d_model, max_seq_len, device)?;
        let dropout = Dropout::new(dropout_rate);

        Ok(Self {
            pe,
            dropout,
            d_model,
            max_seq_len,
        })
    }

    fn create_positional_encoding(
        d_model: usize,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mut pe_data = vec![0.0f32; max_seq_len * d_model];

        for pos in 0..max_seq_len {
            for i in 0..(d_model / 2) {
                let angle = pos as f64 / 10000.0_f64.powf(2.0 * i as f64 / d_model as f64);
                pe_data[pos * d_model + 2 * i] = angle.sin() as f32;
                pe_data[pos * d_model + 2 * i + 1] = angle.cos() as f32;
            }
        }

        Tensor::from_vec(pe_data, (max_seq_len, d_model), device)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 3 {
            return Err(candle_core::Error::Msg(
                "Input tensor must be 3D [batch_size, seq_len, d_model]".to_string(),
            ));
        }

        let (batch_size, seq_len, d_model) = (dims[0], dims[1], dims[2]);

        if d_model != self.d_model {
            return Err(candle_core::Error::Msg(format!(
                "Input d_model {} doesn't match expected {}",
                d_model, self.d_model
            )));
        }

        if seq_len > self.max_seq_len {
            return Err(candle_core::Error::Msg(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.max_seq_len
            )));
        }

        let pe_slice = self.pe.narrow(0, 0, seq_len)?;
        let pe_expanded = pe_slice
            .unsqueeze(0)?
            .expand((batch_size, seq_len, d_model))?;
        let output = x.add(&pe_expanded)?;

        self.dropout.forward(&output, true)
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

impl Embeddings {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        max_seq_len: usize,
        dropout_rate: f32,
        device: &Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(vocab_size, d_model, vb.pp("word_embeddings"))?;
        let positional_encoding =
            PositionalEncoding::new(d_model, max_seq_len, dropout_rate, device)?;

        Ok(Self {
            word_embeddings,
            positional_encoding,
            d_model,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.word_embeddings.forward(input_ids)?;
        let scale = (self.d_model as f32).sqrt();
        let scaled_embeddings = embeddings.mul(&Tensor::new(scale, embeddings.device())?)?;
        self.positional_encoding.forward(&scaled_embeddings)
    }
}
