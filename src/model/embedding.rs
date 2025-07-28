use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{Dropout, Module};
use std::f64::consts::PI;

// Positional Encoding as described in "Attention is All You Need"
// Adds sinusoidal positional information to embeddings
pub struct PositionalEncoding {
    // Pre-computed positional encodings [max_seq_len, d_model]
    pe: Tensor,
    // Dropout layer to apply after adding positional encoding
    dropout: Dropout,
    // Model dimension
    d_model: usize,
    // Maximum sequence length supported
    max_seq_len: usize,
}


impl PositionalEmbedding {
    pub fn new(
        d_model: usize, 
        max_seq_len: usize, 
        dropout_rate: f32,
        device: &Device
    ) -> Result<Self> {
        if d_model % 2 != 0 {
            return Err(candle_core::Error::Msg(
                "d_model must be even for positional encoding".to_string()
            ));
        }

        // Create positional encoding matrix
        let pe = Self::create_positional_encoding(d_model, max_seq_len, device)?;
        let dropout = Dropout::new(dropout_rate);

        Ok(Self {
            pe,
            dropout,
            d_model,
            max_seq_len,
        })
    }

    // Creates the sinusoidal positional encoding matrix
    // 
    // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    fn create_positional_encoding(
        d_model: usize, 
        max_seq_len: usize, 
        device: &Device
    ) -> Result<Tensor> {
        let mut pe_data = vec![0.0f32; max_seq_len * d_model];
        
        for pos in 0..max_seq_len {
            for i in 0..(d_model / 2) {
                let angle = pos as f64 / 10000.0_f64.powf(2.0 * i as f64 / d_model as f64);
                // Even indices: sin
                pe_data[pos * d_model + 2 * i] = angle.sin() as f32;
                // Odd indices: cos  
                pe_data[pos * d_model + 2 * i + 1] = angle.cos() as f32;
            }
        }

        let pe = Tensor::from_vec(pe_data, (max_seq_len, d_model), device)?;
        Ok(pe)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 3 {
            return Err(candle_core::Error::Msg(
                "Input tensor must be 3D [batch_size, seq_len, d_model]".to_string()
            ));
        }

        let (batch_size, seq_len, d_model) = (dims[0], dims[1], dims[2]);
        
        if d_model != self.d_model {
            return Err(candle_core::Error::Msg(
                format!("Input d_model {} doesn't match expected {}", d_model, self.d_model)
            ));
        }

        if seq_len > self.max_seq_len {
            return Err(candle_core::Error::Msg(
                format!("Sequence length {} exceeds maximum {}", seq_len, self.max_seq_len)
            ));
        }

        // Get positional encodings for the current sequence length
        let pe_slice = self.pe.narrow(0, 0, seq_len)?; // [seq_len, d_model]
        
        // Expand to match batch size: [1, seq_len, d_model] -> [batch_size, seq_len, d_model]
        let pe_expanded = pe_slice.unsqueeze(0)?.expand((batch_size, seq_len, d_model))?;
        
        // Add positional encoding to input
        let output = x.add(&pe_expanded)?;
        
        // Apply dropout
        self.dropout.forward(&output, true)
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}
