import torch
from transformers import AutoTokenizer
from datasets import load_dataset

wmt_dataset = load_dataset('wmt14', 'fr-en')
tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)

def tokenize(examples):
    english_examples = [example['en'] for example in examples['translation']]
    french_examples = [example['fr'] for example in examples['translation']]
    
    english_examples = tokenizer(english_examples, padding='max_length', truncation=True, max_length=128)
    french_examples = tokenizer(french_examples, padding='max_length', truncation=True, max_length=128)
    return {
        # all of these should have shape (batch_size, max_length)
        'input_token_ids': french_examples['input_ids'], 
        'encoder_attention_mask': french_examples['attention_mask'], # mask for padded sequences
        'decoder_attention_mask': english_examples['attention_mask'],
        'labels': english_examples['input_ids']
    }


