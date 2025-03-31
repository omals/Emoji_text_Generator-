import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import EmojiEnhancer
from src.model import EmojiTransformer 
from flask import Flask, request, render_template, jsonify
import json

app = Flask(__name__)

# Load emoji mappings
with open('data/emoji_mappings/word_emoji_map.json') as f:
    word_emoji_map = json.load(f)
with open('data/emoji_mappings/sentiment_emoji_map.json') as f:
    sentiment_emoji_map = json.load(f)

# Initialize enhancer
enhancer = EmojiEnhancer(
    model_path='models/emoji_model.pth',
    tokenizer_path='bert-base-uncased',
    word_emoji_map=word_emoji_map,
    sentiment_emoji_map=sentiment_emoji_map
)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text', '')
        result = enhancer.enhance_text(text)
        return render_template('index.html', 
                             original_text=result['original_text'],
                             enhanced_text=result['enhanced_text'],
                             sentiment_emoji=result['sentiment_emoji'])
    
    return render_template('index.html')

@app.route('/api/enhance', methods=['POST'])
def api_enhance():
    data = request.get_json()
    text = data.get('text', '')
    result = enhancer.enhance_text(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port = 5001)