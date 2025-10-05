"""
Transformer Model Training Script for Machine Translation

WHAT THIS CODE DOES:
This script trains a transformer model for machine translation between two languages.
It implements the complete training pipeline from data loading to model saving, following
the transformer architecture described in "Attention Is All You Need" paper.

WHY THIS TRAINING PROCESS IS IMPORTANT:
Machine translation requires a sophisticated training process because:
1. The model needs to learn complex language patterns and relationships
2. Different languages have different vocabularies, grammar, and structures
3. The model must learn to encode meaning from source language and decode to target language
4. Training requires careful data preparation, tokenization, and validation

KEY TRAINING COMPONENTS EXPLAINED:

TOKENIZATION:
- Converts text into numerical tokens that the model can process
- Creates vocabulary mappings for both source and target languages
- Handles unknown words and special tokens ([SOS], [EOS], [PAD], [UNK])

DATA PREPARATION:
- Loads bilingual sentence pairs from datasets
- Splits data into training and validation sets
- Creates DataLoaders for efficient batch processing
- Ensures consistent sequence lengths through padding

MODEL TRAINING LOOP:
- Forward pass: Encoder processes source → Decoder generates target
- Loss calculation: Compares predicted vs actual target sentences
- Backward pass: Updates model weights using gradients
- Validation: Tests model performance on unseen data

GREEDY DECODING:
- Inference method that generates translations word by word
- Always picks the most likely next word (greedy approach)
- Used during validation to see actual translation quality

CHECKPOINTING:
- Saves model state after each epoch for resuming training
- Allows loading pre-trained models to continue training
- Prevents loss of progress if training is interrupted

MONITORING:
- TensorBoard logging for tracking training progress
- Validation examples to see translation quality
- Loss tracking to monitor learning progress

This script orchestrates all these components to train an effective translation model.
"""

import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm
import os
import warnings
from pathlib import Path

from dataset import BilingualDataset, causal_mask
from model import BuildTransformer
from config import get_config, get_weights_file_path, latest_weights_file_path



def get_all_sentences(ds, lang):
    """
    Generator function that extracts all sentences in a specific language from the dataset.
    
    WHAT IT DOES:
    - Iterates through the dataset and yields sentences one by one
    - Extracts sentences for the specified language (source or target)
    
    WHY IT'S IMPORTANT:
    - Memory efficient: Uses generator pattern to avoid loading all sentences at once
    - Required for tokenizer training: Tokenizer needs to see all text to build vocabulary
    - Language-specific: Extracts only sentences in the language we're building tokenizer for
    
    HOW IT WORKS:
    - Takes dataset and language code as input
    - For each item in dataset, extracts the translation for specified language
    - Yields sentences one at a time (generator pattern)
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    Creates or loads a tokenizer for the specified language.
    
    WHAT IT DOES:
    - Checks if tokenizer already exists, loads it if found
    - If not found, builds a new tokenizer from the dataset
    - Saves the newly built tokenizer for future use
    
    WHY IT'S IMPORTANT:
    - Tokenizers convert text to numbers that neural networks can process
    - Each language needs its own tokenizer due to different vocabularies
    - Caching prevents rebuilding tokenizer every time (expensive operation)
    - WordLevel tokenizer treats each word as a separate token
    
    HOW IT WORKS:
    1. Check if tokenizer file exists at specified path
    2. If exists: Load and return existing tokenizer
    3. If not exists:
       - Create WordLevel tokenizer with [UNK] token for unknown words
       - Set pre-tokenizer to split on whitespace
       - Define special tokens: [UNK], [PAD], [SOS], [EOS]
       - Train tokenizer on all sentences in the language
       - Save tokenizer to file for future use
    4. Return the tokenizer
    
    SPECIAL TOKENS EXPLAINED:
    - [UNK]: Unknown words not in vocabulary
    - [PAD]: Padding to make sequences same length
    - [SOS]: Start of sentence marker
    - [EOS]: End of sentence marker
    """
    # config['tokenizer_file'] == '../tokenizers/tokenizer{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Build new tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # Load existing tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_dataset(config):
    """
    Loads and prepares the dataset for training and validation.
    
    WHAT IT DOES:
    - Loads bilingual translation dataset from HuggingFace
    - Creates tokenizers for both source and target languages
    - Splits data into training and validation sets
    - Creates PyTorch DataLoaders for efficient batch processing
    - Analyzes maximum sentence lengths for debugging
    
    WHY IT'S IMPORTANT:
    - Data preparation is crucial for model training success
    - Proper train/validation split prevents overfitting
    - DataLoaders enable efficient batch processing and GPU utilization
    - Tokenizers convert text to numerical format the model can understand
    
    HOW IT WORKS:
    1. Load raw dataset from 'opus_books' (book translations)
    2. Create/load tokenizers for source and target languages
    3. Split dataset: 90% training, 10% validation
    4. Wrap raw data with BilingualDataset class (adds padding, masks, etc.)
    5. Calculate max sentence lengths for analysis
    6. Create DataLoaders with appropriate batch sizes
    7. Return everything needed for training
    
    BATCH SIZE CONSIDERATIONS:
    - Training: Uses configured batch_size for efficiency
    - Validation: Uses batch_size=1 for easier analysis of individual examples
    """
    # Load the raw bilingual dataset
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build or load tokenizers for both languages
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Split dataset: 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size  # Ensure they sum to total length
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Wrap with BilingualDataset to add tokenization, padding, and masking
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Analyze maximum sentence lengths (useful for debugging and optimization)
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Create DataLoaders for efficient batch processing
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)  # batch_size=1 for validation

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Creates and initializes the transformer model.
    
    WHAT IT DOES:
    - Instantiates a transformer model with specified configuration
    - Sets vocabulary sizes for source and target languages
    - Configures model dimensions and sequence length
    
    WHY IT'S IMPORTANT:
    - The model architecture determines translation quality
    - Vocabulary sizes must match the tokenizers
    - Sequence length must match the dataset preparation
    - Model dimensions affect capacity and training time
    
    HOW IT WORKS:
    - Calls BuildTransformer constructor with:
      * vocab_src_len: Size of source language vocabulary
      * vocab_tgt_len: Size of target language vocabulary
      * seq_len: Maximum sequence length (same for encoder and decoder)
      * d_model: Model dimension (embedding size)
    - Returns initialized model ready for training
    """
    model = BuildTransformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Generates translation using greedy decoding strategy.
    
    WHAT IT DOES:
    - Takes a source sentence and generates target translation word by word
    - Uses "greedy" approach: always picks the most likely next word
    - Continues until it generates [EOS] token or reaches maximum length
    
    WHY IT'S IMPORTANT:
    - This is how we actually use the trained model for translation
    - Greedy decoding is fast and simple (though not always optimal)
    - Used during validation to see actual translation quality
    - Demonstrates the model's learned translation capabilities
    
    HOW IT WORKS:
    1. Get special token IDs for start-of-sentence and end-of-sentence
    2. Encode the source sentence once (reuse for all decoding steps)
    3. Initialize decoder input with [SOS] token
    4. Generate translation word by word:
       - Create causal mask (prevents looking at future words)
       - Run decoder to get probability distribution over vocabulary
       - Pick word with highest probability (greedy choice)
       - Add chosen word to decoder input
       - Repeat until [EOS] or max length reached
    5. Return the generated sequence
    
    GREEDY vs OTHER STRATEGIES:
    - Greedy: Fast, deterministic, but may miss better translations
    - Beam Search: Slower but explores multiple possibilities
    - Sampling: Introduces randomness for more diverse outputs
    
    CAUSAL MASKING:
    - Ensures decoder can only see previous words, not future ones
    - Critical for proper autoregressive generation
    - Prevents the model from "cheating" during inference
    """
    # Get special token indices
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every decoding step
    # This is efficient because encoder output doesn't change during decoding
    encoder_output = model.encode(source, source_mask)
    
    # Initialize decoder input with start-of-sentence token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    # Generate translation word by word
    while True:
        # Stop if we've reached maximum length
        if decoder_input.size(1) == max_len:
            break

        # Create causal mask for current decoder input length
        # This prevents the decoder from seeing future tokens
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Run decoder to get output representations
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Project to vocabulary size and get probabilities for next token
        prob = model.project(out[:, -1])  # Only look at the last position
        
        # Greedy choice: pick the token with highest probability
        _, next_word = torch.max(prob, dim=1)
        
        # Add the chosen word to decoder input for next iteration
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        # Stop if we generated end-of-sentence token
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)  # Remove batch dimension


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    """
    Runs validation to evaluate model performance on unseen data.
    
    WHAT IT DOES:
    - Tests the model on validation examples without updating weights
    - Generates actual translations using greedy decoding
    - Displays source, target, and predicted translations for comparison
    - Collects examples for potential metric calculation
    
    WHY IT'S IMPORTANT:
    - Validation prevents overfitting by testing on unseen data
    - Shows actual translation quality, not just loss numbers
    - Helps monitor if the model is learning meaningful patterns
    - Provides human-readable feedback during training
    - Essential for determining when to stop training
    
    HOW IT WORKS:
    1. Set model to evaluation mode (disables dropout, etc.)
    2. Get console width for pretty printing
    3. For each validation example:
       - Extract encoder input and mask
       - Use greedy decoding to generate translation
       - Convert tokens back to human-readable text
       - Display source, target, and predicted translations
       - Stop after specified number of examples
    4. All operations done with torch.no_grad() for efficiency
    
    EVALUATION MODE:
    - model.eval() switches to evaluation mode
    - Disables dropout and batch normalization updates
    - Ensures consistent behavior during validation
    
    NO_GRAD CONTEXT:
    - Disables gradient computation for efficiency
    - Reduces memory usage during validation
    - Speeds up inference since we don't need gradients
    """
    # Set model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()
    count = 0

    # Lists to store validation examples (could be used for metrics)
    source_texts = []
    expected = []
    predicted = []

    try:
        # Get console window width for pretty printing
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    # Disable gradient computation for efficiency during validation
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            # Move inputs to device
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # Validation uses batch_size=1 for easier analysis
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Generate translation using greedy decoding
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # Extract human-readable texts
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Store for potential metric calculation
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Display comparison of source, target, and predicted translations
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            # Only show a few examples to avoid cluttering output
            if count == num_examples:
                print_msg('-'*console_width)
                break


def train_model(config):
    """
    Main training function that orchestrates the entire training process.
    
    WHAT IT DOES:
    - Sets up the training environment (device, data, model)
    - Implements the training loop with forward/backward passes
    - Handles model checkpointing and resuming training
    - Runs validation after each epoch
    - Logs training progress to TensorBoard
    
    WHY IT'S IMPORTANT:
    - This is the core function that actually trains the transformer model
    - Implements the complete training pipeline from scratch
    - Handles all the complex details of neural network training
    - Ensures proper model saving and loading for long training runs
    - Provides monitoring and validation to track progress
    
    HOW IT WORKS:
    1. SETUP PHASE:
       - Detect and configure GPU/CPU device
       - Load dataset and create data loaders
       - Initialize model, optimizer, and loss function
       - Set up TensorBoard logging
       - Load pre-trained model if specified
    
    2. TRAINING LOOP (for each epoch):
       - Set model to training mode
       - For each batch:
         * Forward pass: encoder → decoder → projection
         * Calculate loss comparing predictions to targets
         * Backward pass: compute gradients
         * Update model weights
         * Log progress
       - Run validation to check performance
       - Save model checkpoint
    
    3. KEY TRAINING CONCEPTS:
       - Teacher Forcing: Decoder sees correct previous tokens during training
       - Cross Entropy Loss: Measures prediction accuracy
       - Adam Optimizer: Adaptive learning rate optimization
       - Label Smoothing: Prevents overconfident predictions
       - Gradient Accumulation: Updates weights after processing batch
    """
    
    # =============================================================================
    # DEVICE SETUP: Configure GPU/CPU for training
    # =============================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Display device information for debugging and optimization
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # =============================================================================
    # DATA AND MODEL SETUP
    # =============================================================================
    # Create directory for saving model weights
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Load dataset and create data loaders
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    
    # Initialize model and move to device (GPU/CPU)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Set up TensorBoard for logging training metrics
    writer = SummaryWriter(config['experiment_name'])

    # Initialize Adam optimizer with specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # =============================================================================
    # MODEL LOADING: Resume training from checkpoint if specified
    # =============================================================================
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # =============================================================================
    # LOSS FUNCTION: CrossEntropy with label smoothing and padding ignore
    # =============================================================================
    # ignore_index: Don't calculate loss for padding tokens
    # label_smoothing: Prevents overconfident predictions, improves generalization
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # =============================================================================
    # MAIN TRAINING LOOP
    # =============================================================================
    for epoch in range(initial_epoch, config['num_epochs']):
        # Clear GPU cache to prevent memory issues
        torch.cuda.empty_cache()
        
        # Set model to training mode (enables dropout, batch norm updates)
        model.train()
        
        # Create progress bar for this epoch
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        # Process each batch in the training data
        for batch in batch_iterator:
            # Move batch data to device (GPU/CPU)
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # =============================================================================
            # FORWARD PASS: Run data through the transformer model
            # =============================================================================
            # 1. Encoder processes source language input
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            
            # 2. Decoder processes target language input with encoder context
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            
            # 3. Project decoder output to vocabulary size for word predictions
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # =============================================================================
            # LOSS CALCULATION: Compare predictions with actual target words
            # =============================================================================
            label = batch['label'].to(device) # (B, seq_len)

            # Calculate cross-entropy loss between predictions and labels
            # Reshape to (B*seq_len, vocab_size) and (B*seq_len) for loss calculation
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            # Update progress bar with current loss
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log loss to TensorBoard for monitoring
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # =============================================================================
            # BACKWARD PASS: Update model weights based on loss
            # =============================================================================
            # Calculate gradients (how to adjust weights to reduce loss)
            loss.backward()

            # Update model weights using calculated gradients
            optimizer.step()
            
            # Clear gradients for next iteration (PyTorch accumulates by default)
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # =============================================================================
        # VALIDATION: Test model performance after each epoch
        # =============================================================================
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # =============================================================================
        # CHECKPOINTING: Save model state after each epoch
        # =============================================================================
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)


    




