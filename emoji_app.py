# import emoji
# from difflib import get_close_matches
# import re

# def get_all_emoji_aliases():
#     """Get all emoji names and aliases from the emoji library."""
#     emoji_aliases = {}
#     for emoji_code in emoji.EMOJI_DATA.keys():
#         data = emoji.EMOJI_DATA[emoji_code]
#         # Get the main name and all aliases
#         aliases = []
#         if 'en' in data:
#             aliases.extend(data['en'].lower().replace('_', ' ').split())
#         if 'alias' in data:
#             aliases.extend([alias.lower().replace('_', ' ') for alias in data['alias']])
        
#         # Store the emoji with all its aliases
#         for alias in aliases:
#             if alias not in emoji_aliases:
#                 emoji_aliases[alias] = []
#             emoji_aliases[alias].append(emoji_code)
    
#     return emoji_aliases

# def predict_emojis(word, emoji_aliases, max_predictions=5):
#     """Predict emojis based on input word."""
#     word = word.lower().strip()
#     predictions = set()
    
#     # Direct matches
#     for alias, emoji_codes in emoji_aliases.items():
#         if word in alias or alias in word:
#             predictions.update(emoji_codes)
    
#     # Find close matches
#     all_aliases = list(emoji_aliases.keys())
#     close_matches = get_close_matches(word, all_aliases, n=3, cutoff=0.6)
#     for match in close_matches:
#         predictions.update(emoji_aliases[match])
    
#     # Convert emoji codes to actual emojis
#     emoji_results = [emoji.emojize(code) for code in predictions]
    
#     # If no predictions, try breaking compound words
#     if not emoji_results:
#         words = re.findall(r'[a-z]+', word)
#         for subword in words:
#             for alias, emoji_codes in emoji_aliases.items():
#                 if subword in alias or alias in subword:
#                     emoji_results.extend([emoji.emojize(code) for code in emoji_codes])
    
#     # Remove duplicates and limit results
#     return list(dict.fromkeys(emoji_results))[:max_predictions]

# def main():
#     print("📱 Loading Emoji Predictor...")
#     emoji_aliases = get_all_emoji_aliases()
#     print("✨ Welcome to Smart Emoji Predictor!")
#     print("Type 'quit' to exit")
    
#     while True:
#         word = input("\n🔍 Enter a word: ").strip()
        
#         if word.lower() == 'quit':
#             print("👋 Goodbye!")
#             break
        
#         if not word:
#             print("❗ Please enter a word!")
#             continue
        
#         predictions = predict_emojis(word, emoji_aliases)
        
#         if predictions:
#             print(f"\n🎯 Predictions for '{word}':")
#             print(" ".join(predictions))
#             print(f"Found {len(predictions)} matching emojis!")
#         else:
#             print(f"🤔 No direct matches found for '{word}'")
#             print("Try a different word or add more context!")

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n\n👋 Goodbye!")

from flask import Flask, request, jsonify, render_template
import emoji
from difflib import get_close_matches
import re

app = Flask(__name__)

def get_all_emoji_aliases():
    """Get all emoji names and aliases from the emoji library."""
    emoji_aliases = {}
    for emoji_code in emoji.EMOJI_DATA.keys():
        data = emoji.EMOJI_DATA[emoji_code]
        aliases = []
        if 'en' in data:
            aliases.extend(data['en'].lower().replace('_', ' ').split())
        if 'alias' in data:
            aliases.extend([alias.lower().replace('_', ' ') for alias in data['alias']])
        
        for alias in aliases:
            if alias not in emoji_aliases:
                emoji_aliases[alias] = []
            emoji_aliases[alias].append(emoji_code)
    
    return emoji_aliases

def predict_emojis(word, emoji_aliases, max_predictions=8):
    """Predict emojis based on input word."""
    word = word.lower().strip()
    predictions = set()
    
    # Direct matches
    for alias, emoji_codes in emoji_aliases.items():
        if word in alias or alias in word:
            predictions.update(emoji_codes)
    
    # Find close matches
    all_aliases = list(emoji_aliases.keys())
    close_matches = get_close_matches(word, all_aliases, n=3, cutoff=0.6)
    for match in close_matches:
        predictions.update(emoji_aliases[match])
    
    # Convert emoji codes to actual emojis
    emoji_results = [emoji.emojize(code) for code in predictions]
    
    # If no predictions, try breaking compound words
    if not emoji_results:
        words = re.findall(r'[a-z]+', word)
        for subword in words:
            for alias, emoji_codes in emoji_aliases.items():
                if subword in alias or alias in subword:
                    emoji_results.extend([emoji.emojize(code) for code in emoji_codes])
    
    # Remove duplicates and limit results
    return list(dict.fromkeys(emoji_results))[:max_predictions]

# Initialize emoji aliases
EMOJI_ALIASES = get_all_emoji_aliases()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    word = data.get('word', '').strip()
    
    if not word:
        return jsonify({'emojis': []})
    
    predictions = predict_emojis(word, EMOJI_ALIASES)
    return jsonify({'emojis': predictions})

if __name__ == '__main__':
    app.run(debug=True)