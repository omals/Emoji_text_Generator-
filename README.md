# Emoji Text Enhancer ✨

Transform your boring text into fun emoji-filled messages! The Emoji Text Enhancer intelligently suggests relevant emojis based on the meaning and sentiment of your text, making your messages more expressive and engaging.

![Emoji Text Enhancer](/images/website.png)

## Features 🚀
- **Intelligent Emoji Suggestions:** Adds emojis based on sentiment and word meaning.
- **Multilingual Support:** Works seamlessly with texts in various languages.
- **Real-Time Enhancement:** Instantaneously enhance your input text.
- **Web Interface:** Simple and intuitive interface built with Flask.
- **Deep Learning Based:** Uses advanced NLP models to understand context and sentiment.

## Installation 🛠️

### Prerequisites
- Python 3.x
- PyTorch
- Flask
- TensorFlow (if applicable)
- Other dependencies (listed in `requirements.txt`)

### Setup Environment
1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Process Data and Create Mappings
1. Create word-emoji mappings:
   ```bash
   python data/scripts/create_word_emoji_map.py
   ```
2. Create sentiment-emoji mappings:
   ```bash
   python data/scripts/create_sentiment_emoji_map.py
   ```
3. Split the dataset:
   ```bash
   python data/scripts/split_dataset.py
   ```

### Train the Model
1. Run the training script:
   ```bash
   python src/train.py
   ```

### Run the Website
1. Start the web application:
   ```bash
   python app/app.py
   ```
2. Open your browser and visit `http://127.0.0.1:5000`

## Project Structure 📂
```
emoji-text-generator/
│
├── data/                       # Dataset files
│   ├── raw/                    # Raw datasets (e.g., Twitter, Sentiment140)
│   │   ├── tweets.csv
│   │   └── sentiment140.csv
│   │
│   ├── processed/              # Processed datasets (train/val/test splits)
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
|   |
│   ├── scripts/   
│   │   └── create_word_emoji_map.py  # The processing code goes here
│   │
│   └── emoji_mappings/         # Emoji mappings (word → emoji, sentiment → emoji)
│       ├── word_emoji_map.json
│       └── sentiment_emoji_map.json
│
├── models/                     # Saved models
│   ├── emoji_model.pth         # Trained PyTorch model
│   └── tokenizer/              # Saved tokenizer (BERT)
│
├── src/                        # Source code
│   ├── data_processing.py      # Dataset loading & preprocessing
│   ├── model.py                # EmojiTransformer model definition
│   ├── train.py                # Training script
│   ├── inference.py            # EmojiEnhancer class for predictions
│   └── utils.py                # Helper functions (emoji mappings, etc.)
│
├── app/                        # Flask web application
│   ├── static/                 # CSS/JS (if needed)
│   ├── templates/              # HTML templates
│   │   └── index.html
│   └── app.py                  # Flask server
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # For containerization (optional)
└── README.md                   # Project documentation
```

## Usage 💡
1. Open the web app.
2. Enter your text in the input box.
3. Click on the 'Add Emojis' button.
4. Get your enhanced text with emojis!

## Contributing 🤝
Feel free to open issues and submit pull requests. Your contributions are welcome!

## License 📝
This project is licensed under the MIT License.


