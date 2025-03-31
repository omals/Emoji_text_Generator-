import pandas as pd
import emoji
from collections import defaultdict
import json
import re
import os

# Configure paths
RAW_DATA_PATH = os.path.join('data', 'raw', 'tweets.csv')
OUTPUT_PATH = os.path.join('data', 'emoji_mappings', 'word_emoji_map.json')

# Create directories if they don't exist
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load data
try:
    tweets_df = pd.read_csv(RAW_DATA_PATH)
except FileNotFoundError:
    print(f"Error: Input file not found at {RAW_DATA_PATH}")
    exit()

# Processing logic
word_emoji_counts = defaultdict(lambda: defaultdict(int))

for _, row in tweets_df.iterrows():
    emoji_char = row['Emoji']
    context = str(row['Context'])
    
    context = re.sub(r'http\S+|@\S+|[^\w\s]', '', context)
    words = [w.lower() for w in context.split() if w.isalpha()]
    
    for word in words:
        if emoji_char in emoji.EMOJI_DATA:
            word_emoji_counts[word][emoji_char] += 1

word_emoji_map = {}
for word, emoji_counts in word_emoji_counts.items():
    filtered = {e: cnt for e, cnt in emoji_counts.items() if cnt >= 2}
    if filtered:
        top_emojis = sorted(filtered.items(), key=lambda x: -x[1])[:3]
        word_emoji_map[word] = [e[0] for e in top_emojis]

# Save output
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(word_emoji_map, f, ensure_ascii=False)

print(f"Successfully generated {OUTPUT_PATH}")
print(f"Total word-emoji mappings: {len(word_emoji_map)}")


# Positive Emotions / Reactions
# Love / Affection: ❤️, 😘, 💕, 🥰, 😍

# Happy / Joy: 😊, 😂, 🤣, 😄, 🙌

# Celebration / Excitement: 🎉, 🤩, 🙏, 👏

# Support / Encouragement: 🤗, 🙌, 💪

# Negative Emotions / Reactions
# Angry / Annoyed: 😡, 😠, 😤, 🙄, 😒

# Sad / Heartbroken: 😢, 😭, 💔, 😔

# Confusion / Thinking: 🤔, 😳, 😕

# Neutral / Miscellaneous
# Surprise / Shock: 😳, 😲, 😮

# Cool / Confident: 😎, 😏, 🆒

# Funny / Playful: 😜, 😆, 🤪

# Sarcasm / Skepticism: 😏, 🙄