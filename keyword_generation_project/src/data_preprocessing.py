from transformers import BertTokenizer
import pandas as pd

def preprocess_data(text_data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def clean_text(text):
        # Basic text cleaning
        text = text.lower().strip()
        return text
    
    processed_data = [clean_text(text) for text in text_data]
    return processed_data