import json
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch

class EmojiDataset(Dataset):
    def __init__(self, texts, word_emoji_labels, sentiment_labels, sentiment_emoji_labels, tokenizer, max_length=128):
        self.texts = texts
        self.word_emoji_labels = word_emoji_labels
        self.sentiment_labels = sentiment_labels
        self.sentiment_emoji_labels = sentiment_emoji_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        word_emoji_label = self.word_emoji_labels[idx]
        sentiment_label = self.sentiment_labels[idx]
        sentiment_emoji_label = self.sentiment_emoji_labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Pad word emoji labels
        padded_word_emoji_labels = torch.zeros(self.max_length, dtype=torch.long)
        word_emoji_label_tensor = torch.tensor(word_emoji_label[:self.max_length])
        padded_word_emoji_labels[:len(word_emoji_label_tensor)] = word_emoji_label_tensor
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'word_emoji_labels': padded_word_emoji_labels,
            'sentiment_labels': torch.tensor(sentiment_label, dtype=torch.long),
            'sentiment_emoji_labels': torch.tensor(sentiment_emoji_label, dtype=torch.long)
        }

def load_emoji_mappings():
    with open('../data/emoji_mappings/word_emoji_map.json') as f:
        word_emoji_map = json.load(f)
    with open('../data/emoji_mappings/sentiment_emoji_map.json') as f:
        sentiment_emoji_map = json.load(f)
    return word_emoji_map, sentiment_emoji_map