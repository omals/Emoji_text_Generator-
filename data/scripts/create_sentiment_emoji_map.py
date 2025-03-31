import pandas as pd
import json
from collections import defaultdict
import chardet

import pandas as pd
import re

# Load Sentiment140 dataset (adjust path/encoding as needed)
df = pd.read_csv('data/raw/sentiment140.csv', encoding='latin1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Extract emojis from text (simplified regex)
def extract_emojis(text):
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]'
    )
    return ' '.join(emoji_pattern.findall(str(text)))

df['emojis'] = df['text'].apply(extract_emojis)

# Filter binary sentiment (ignore neutral)
df = df[df['target'].isin([0, 4])].copy()
df['sentiment'] = df['target'].replace({0: 'negative', 4: 'positive'})

# Now use your original code to map emojis to sentiment
# # Detect file encoding
# with open('data/raw/sentiment140.csv', 'rb') as f:
#     encoding = chardet.detect(f.read())['encoding']

# # Load CSV with detected encoding (fallback to 'latin1')
# try:
#     df = pd.read_csv('data/raw/sentiment140.csv', encoding=encoding)
# except UnicodeDecodeError:
#     df = pd.read_csv('data/raw/sentiment140.csv', encoding='latin1')

# df = pd.read_csv('data/raw/sentiment140.csv', encoding='latin1')  # Fallback for most cases

# Proceed with your analysis
sentiment_emoji_counts = defaultdict(lambda: defaultdict(int))
print(df.head())
for _, row in df.iterrows():
    sentiment = 'positive' if row['sentiment'] == 1 else 'negative'
    for emoji in set(row['emojis'].split()):  # Remove duplicates
        sentiment_emoji_counts[sentiment][emoji] += 1
print(sentiment_emoji_counts)
# Save results
sentiment_emoji_map = {
    'positive': [e[0] for e in sorted(sentiment_emoji_counts['positive'].items(), key=lambda x: -x[1])[:5]],
    'negative': [e[0] for e in sorted(sentiment_emoji_counts['negative'].items(), key=lambda x: -x[1])[:5]],
    'neutral': ["😐", "🤷", "✋", "🚶", "💤"]
}

with open('data/emoji_mappings/sentiment_emoji_map.json', 'w', encoding='utf-8') as f:
    json.dump(sentiment_emoji_map, f, ensure_ascii=False)