import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import emoji

class EmojiTransformer(nn.Module):
    def __init__(self, num_word_emojis, num_sentiment_emojis, model_name='bert-base-uncased',
             word_dropout=0.3, sentiment_dropout=0.2, temperature=1.5):
        self.temperature = temperature
        super(EmojiTransformer, self).__init__()
        
        # Base transformer model
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Word emoji prediction head
        self.word_emoji_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, num_word_emojis)
            )
        # Add proper weight initialization
        for layer in self.word_emoji_head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

        # Sentence sentiment emoji prediction head
        self.sentiment_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 5)  # 5 sentiment classes
        )
        
        # Sentiment emoji prediction
        self.sentiment_emoji_head = nn.Linear(5, num_sentiment_emojis)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Word-level features (last hidden state)
        word_features = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Sentence-level features (pooled output)
        sentence_features = outputs.pooler_output  # [batch_size, hidden_size]
        
        # Word emoji predictions
        word_emoji_logits = self.word_emoji_head(word_features)  # [batch_size, seq_len, num_word_emojis]
        
        # Sentiment predictions
        sentiment_logits = self.sentiment_head(sentence_features)  # [batch_size, 5]
        
        # Sentiment emoji predictions
        sentiment_emoji_logits = self.sentiment_emoji_head(sentiment_logits)  # [batch_size, num_sentiment_emojis]
        
        return word_emoji_logits, sentiment_logits, sentiment_emoji_logits