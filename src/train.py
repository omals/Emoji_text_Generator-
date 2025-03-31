import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import EmojiTransformer
from data_processing import EmojiDataset, load_emoji_mappings
import json

def train_model():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load sample data (replace with your actual data loading)
    texts = ["I love pizza", "This movie was bad"]
    word_emoji_labels = [[0, 1, 2], [0, 0, 0, 3]]  # Sample labels
    sentiment_labels = [0, 1]  # 0=positive, 1=negative
    sentiment_emoji_labels = [0, 1]  # Corresponding emoji labels
    
    # Create dataset
    dataset = EmojiDataset(texts, word_emoji_labels, sentiment_labels, sentiment_emoji_labels, tokenizer)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    model = EmojiTransformer(num_word_emojis=99, num_sentiment_emojis=35).to(device)
    
     
    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    word_criterion = nn.CrossEntropyLoss(ignore_index=-100,label_smoothing=0.1)  # Ignore padding
    sentiment_criterion = nn.CrossEntropyLoss()
    sentiment_emoji_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(80):  # Increased epochs
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word_labels = batch['word_emoji_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)
            sentiment_emoji_labels = batch['sentiment_emoji_labels'].to(device)
            
            word_logits, sentiment_logits, sentiment_emoji_logits = model(input_ids, attention_mask)
            
            # Calculate losses
            word_loss = word_criterion(
                word_logits.view(-1, word_logits.size(-1)), 
                word_labels.view(-1)
            )

            print(f"Word loss: {word_loss.item()}")
            
            sentiment_loss = sentiment_criterion(sentiment_logits, sentiment_labels)
            sentiment_emoji_loss = sentiment_emoji_criterion(sentiment_emoji_logits, sentiment_emoji_labels)
            
            loss = word_loss + sentiment_loss + sentiment_emoji_loss
            loss.backward()
            # GRADIENT CLIPPING GOES HERE
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/emoji_model.pth')
    print("Model saved")

if __name__ == "__main__":
    train_model()