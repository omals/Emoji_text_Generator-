import sys
from transformers import BertTokenizer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import EmojiTransformer
import json
import torch
import random
import numpy as np

class EmojiEnhancer:
    def __init__(self, model_path, tokenizer_path, word_emoji_map, sentiment_emoji_map):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store mappings
        self.word_emoji_map = word_emoji_map  # {"word": "😊", ...}
        self.sentiment_emoji_map = sentiment_emoji_map  # {"happy": ["😊",...], ...}
        
        # Create vocabulary lists
        self.word_vocab = list(word_emoji_map.keys())
        self.sentiment_vocab = list(sentiment_emoji_map.keys())
        
        # Initialize model
        self.model = EmojiTransformer(
            num_word_emojis=len(self.word_vocab),
            num_sentiment_emojis=len(self.sentiment_vocab)
        ).to(self.device)
        
        # Load model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        # Common words to skip emojis for
        self.skip_words = {'the', 'a', 'an', 'in', 'of', 'at', 'on', 'and', 'or', 'is'}
    
    def enhance_text(self, text):
        # First get sentence-level sentiment
        sentiment_emoji = self._get_sentiment_emoji(text)
        
        # Then enhance words
        enhanced_words = []
        for word in text.split():
            lower_word = word.lower().strip(",.!?")
            
            # Skip common words
            if lower_word in self.skip_words:
                enhanced_words.append(word)
                continue
                
            # Get word emoji if exists
            emoji = self.word_emoji_map.get(lower_word, "")
            enhanced_words.append(f"{word}{emoji}")
        
        # Combine results
        enhanced_text = ' '.join(enhanced_words) + " " + sentiment_emoji
        
        return {
            'original_text': text,
            'enhanced_text': enhanced_text,
            'sentiment_emoji': sentiment_emoji
        }
    
    def _get_sentiment_emoji(self, text):
        """Predict sentiment and return appropriate emoji"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=128,
            padding='max_length',
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            _, sentiment_logits, _ = self.model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            sentiment_idx = torch.argmax(sentiment_logits).item()
        
        sentiment = self.sentiment_vocab[sentiment_idx]
        return random.choice(self.sentiment_emoji_map[sentiment])