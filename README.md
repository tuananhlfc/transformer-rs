# transformer-rs
An implementation of transformer model in Rust
## Features

- Pure Rust implementation of transformer architectures
- Support for encoder, decoder, and encoder-decoder models
- Multi-head attention, positional encoding, and feed-forward layers
- Configurable hyperparameters
- Easy-to-use API for training and inference

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
transformer-rs = "0.1"
```

## Usage

```rust
use transformer_rs::{Transformer, TransformerConfig};

let config = TransformerConfig::default();
let model = Transformer::new(config);

// Example input
let input = vec![1, 2, 3, 4];
let output = model.forward(&input);
println!("{:?}", output);
```

## Examples

See the [`examples/`](examples/) directory for sample usage and training scripts.

## Documentation

Full API documentation is available [here](https://docs.rs/transformer-rs).

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.
