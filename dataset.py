"""
BilingualDataset: A PyTorch Dataset for Machine Translation

WHAT THIS CODE DOES:
    This file implements a custom PyTorch Dataset class for training transformer models on machine translation tasks.
    It takes pairs of sentences in two different languages (source and target) and prepares them in the specific
    format required by transformer models like the one described in "Attention Is All You Need" paper.

WHY THIS IS IMPORTANT:
    Machine translation models need data in a very specific format:
    1. Text must be converted to numbers (tokens) that the model can process
    2. Special tokens must be added to mark sentence boundaries
    3. All sequences must be the same length (padding)
    4. Different inputs are needed for the encoder and decoder parts of the transformer
    5. Attention masks are needed to tell the model which parts to pay attention to

KEY CONCEPTS EXPLAINED:

    TOKENIZATION:
    - Converts human-readable text into numerical tokens that neural networks can process
    - Each word/subword gets mapped to a unique integer ID

    SPECIAL TOKENS:
    - [SOS] (Start of Sentence): Tells the model where a sentence begins
    - [EOS] (End of Sentence): Tells the model where a sentence ends
    - [PAD] (Padding): Fills empty space to make all sequences the same length

    ENCODER vs DECODER INPUTS:
    - Encoder gets the source language sentence (what we want to translate FROM)
    - Decoder gets the target language sentence (what we want to translate TO)
    - But decoder input and labels are slightly different for training purposes

    ATTENTION MASKS:
    - Tell the model which tokens are real content vs padding
    - Causal mask ensures decoder can only look at previous tokens (not future ones)

    SEQUENCE LENGTH:
    - All inputs must be the same length for efficient batch processing
    - Shorter sentences get padded, longer sentences cause errors

    This dataset prepares all these components so the transformer model can learn to translate
    from source language to target language effectively.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add SOS, EOS and PAD to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # We will add <s> and </s> thats why '-2'
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # We will only add <s>, and </s> only on the label

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim = 0
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0





