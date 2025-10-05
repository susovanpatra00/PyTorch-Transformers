# PyTorch Transformers: Machine Translation Implementation

A complete implementation of the Transformer architecture for machine translation, based on the groundbreaking paper "Attention Is All You Need" by Vaswani et al. This project provides a comprehensive, educational implementation with extensive documentation to help others learn and understand how Transformers work.

## 🎯 What This Project Does

This project implements a full machine translation system using the Transformer architecture. It can translate text between different languages (currently configured for English to French) by learning from bilingual sentence pairs. The implementation includes:

- **Complete Transformer Architecture**: Encoder-decoder model with multi-head attention
- **Training Pipeline**: Full training loop with validation and checkpointing
- **Data Processing**: Tokenization, padding, and masking for efficient batch processing
- **Inference**: Greedy decoding for generating translations
- **Monitoring**: TensorBoard integration for tracking training progress

## 🏗️ Architecture Overview

The Transformer model consists of several key components:

### Core Components
- **Input Embeddings**: Convert tokens to dense vector representations
- **Positional Encoding**: Add position information to embeddings
- **Multi-Head Attention**: Allow the model to focus on different parts of the input
- **Feed Forward Networks**: Process attention outputs
- **Layer Normalization**: Stabilize training
- **Residual Connections**: Enable deep network training

### Model Structure
- **Encoder**: Processes the source language input
- **Decoder**: Generates the target language output
- **Projection Layer**: Maps decoder output to vocabulary probabilities

## 📁 Project Structure

```
PyTorch-Transformers/
├── train.py          # Main training script with comprehensive documentation
├── model.py          # Transformer model implementation
├── dataset.py        # Bilingual dataset handling and preprocessing
├── config.py         # Configuration settings and hyperparameters
├── NOTES.md          # Theoretical explanations and mathematical foundations
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- CUDA-compatible GPU (recommended for training)

### Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd PyTorch-Transformers
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

1. **Train the model**:
```bash
python train.py
```

2. **Monitor training** (optional):
```bash
tensorboard --logdir=runs
```

The training script will:
- Download the OPUS Books dataset automatically
- Create tokenizers for source and target languages
- Train the model with validation after each epoch
- Save model checkpoints for resuming training
- Display translation examples during validation

## ⚙️ Configuration

Modify [`config.py`](config.py) to customize training parameters:

```python
{
    "batch_size": 8,           # Batch size for training
    "num_epochs": 20,          # Number of training epochs
    "lr": 10**-4,             # Learning rate
    "seq_len": 350,           # Maximum sequence length
    "d_model": 512,           # Model dimension
    "lang_src": "en",         # Source language
    "lang_tgt": "fr",         # Target language
    # ... more parameters
}
```

## 📚 Key Learning Concepts

This implementation demonstrates several important concepts:

### 1. **Attention Mechanism**
- Self-attention in encoder and decoder
- Cross-attention between encoder and decoder
- Multi-head attention for different representation subspaces

### 2. **Training Techniques**
- Teacher forcing during training
- Label smoothing for better generalization
- Gradient accumulation and optimization

### 3. **Data Processing**
- Tokenization with special tokens ([SOS], [EOS], [PAD], [UNK])
- Sequence padding for batch processing
- Attention masking to handle variable-length sequences

### 4. **Inference**
- Greedy decoding for translation generation
- Causal masking to prevent looking at future tokens
- Autoregressive generation

## 🔍 File Descriptions

### [`train.py`](train.py)
The main training script containing:
- Complete training loop with forward/backward passes
- Validation with actual translation examples
- Model checkpointing and resuming
- TensorBoard logging
- Extensive documentation explaining each step

### [`model.py`](model.py)
Transformer model implementation including:
- All transformer components (attention, embeddings, etc.)
- Encoder and decoder stacks
- Model initialization with Xavier uniform weights

### [`dataset.py`](dataset.py)
Bilingual dataset handling:
- PyTorch Dataset class for translation pairs
- Tokenization and sequence preparation
- Attention mask generation
- Padding and special token handling

### [`config.py`](config.py)
Configuration management:
- Hyperparameter settings
- File path utilities
- Model checkpoint handling

### [`NOTES.md`](NOTES.md)
Comprehensive theoretical documentation:
- Mathematical foundations of Transformer architecture explained in simple language
- Step-by-step breakdown of each component (embeddings, attention, normalization, etc.)
- Visual examples and intuitive explanations of complex concepts
- Detailed mathematical formulas with clear explanations
- Perfect companion for understanding the theory behind the implementation

## 📊 Training Process

The training process follows these steps:

1. **Data Loading**: Load bilingual sentence pairs from OPUS Books dataset
2. **Tokenization**: Create vocabularies and convert text to tokens
3. **Model Training**: 
   - Forward pass through encoder and decoder
   - Calculate cross-entropy loss
   - Backward pass and weight updates
4. **Validation**: Generate actual translations to monitor progress
5. **Checkpointing**: Save model state after each epoch

## 🎓 Educational Value

This project is designed for learning and includes:

- **Extensive Comments**: Every function and concept is thoroughly documented
- **Clear Structure**: Code is organized for easy understanding
- **Step-by-Step Explanations**: Complex concepts broken down into digestible parts
- **Real Examples**: See actual translations during training
- **Best Practices**: Proper PyTorch patterns and techniques

## 🙏 Acknowledgments

Special thanks to **Umar Jamil** ([GitHub: @hkproj](https://github.com/hkproj)) for his excellent educational content and inspiration for this implementation.

## 📖 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [PyTorch Documentation](https://pytorch.org/docs/) - Framework documentation

## 🛠️ Requirements

- `torch` - PyTorch deep learning framework
- `datasets` - HuggingFace datasets library
- `tokenizers` - Fast tokenization library
- `tqdm` - Progress bars
- `tensorboard` - Training visualization (included with PyTorch)

## 🚀 Future Improvements

Potential enhancements for learning:
- Beam search decoding
- BLEU score evaluation
- Different attention mechanisms
- Model size variations
- Additional language pairs

## 📝 License

This project is for educational purposes. Feel free to use, modify, and learn from it!

---

*This implementation serves as a comprehensive learning resource for understanding Transformer architecture and machine translation. The extensive documentation and clear structure make it ideal for students and practitioners wanting to understand how modern NLP models work.*