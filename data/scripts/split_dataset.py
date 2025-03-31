import pandas as pd
from sklearn.model_selection import train_test_split
import json

# Load raw data with explicit column renaming for Sentiment140
sentiment_df = pd.read_csv(
    'data/raw/sentiment140.csv',
    encoding='latin1',
    header=None,  # No header in the original file
    names=['target', 'id', 'date', 'flag', 'user', 'text']  # Rename columns
)

# Load tweets.csv (assuming it has 'text' column)
tweets_df = pd.read_csv('data/raw/tweets.csv')

# Ensure both datasets have 'text' column
if 'text' not in tweets_df.columns:
    tweets_df = tweets_df.rename(columns={tweets_df.columns[0]: 'text'})

# Load emoji mappings
with open('data/emoji_mappings/word_emoji_map.json') as f:
    word_emoji_map = json.load(f)

# Combine datasets (keep only 'text' column for simplicity)
df = pd.concat([tweets_df[['text']], sentiment_df[['text']]])

# Assign word-level emoji labels
def get_word_emoji_labels(text):
    return [list(word_emoji_map.get(word, [])) for word in str(text).split()]

df['word_emoji_labels'] = df['text'].apply(get_word_emoji_labels)

# Split data (80% train, 10% val, 10% test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save processed data
train_df.to_csv('data/processed/train.csv', index=False)
val_df.to_csv('data/processed/val.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)