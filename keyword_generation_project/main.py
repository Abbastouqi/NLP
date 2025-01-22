import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from peft import LoraConfig, get_peft_model

class KeywordDataset(Dataset):
    def __init__(self, texts, keywords, tokenizer):
        self.texts = texts
        self.keywords = keywords
        self.tokenizer = tokenizer
        
        # Set padding token for GPT-2 tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        keyword = self.keywords[idx]
        
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        return inputs

class KeywordGenerator:
    def __init__(self):
        # Initialize GPT-2 with LoRA
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_model.resize_token_embeddings(len(self.gpt2_tokenizer))
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["c_attn"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.gpt2_model = get_peft_model(self.gpt2_model, lora_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpt2_model.to(self.device)

    def train_model(self, train_dataloader, epochs=3):
        self.gpt2_model.train()
        optimizer = torch.optim.AdamW(self.gpt2_model.parameters(), lr=2e-5)
        
        print(f"Training on device: {self.device}")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_dataloader):
                batch = {k: v.squeeze().to(self.device) for k, v in batch.items()}
                outputs = self.gpt2_model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def generate_keywords(self, text, max_length=50):
        inputs = self.gpt2_tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        outputs = self.gpt2_model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7
        )
        
        generated_keywords = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_keywords

def main():
    # Load your marketing dataset
    print("Loading dataset...")
    df = pd.read_csv('D:/Tauqer(mf22-9)/keyword_generation_project/data/raw_data/marketing_and_product_performance.csv')
    
    # Prepare text and keywords
    texts = df['Common_Keywords'].tolist()
    keywords = df['Common_Keywords'].tolist()
    
    print(f"Dataset loaded with {len(texts)} samples")
    
    # Initialize keyword generator
    print("Initializing keyword generator...")
    generator = KeywordGenerator()
    
    # Create dataset and dataloader
    dataset = KeywordDataset(texts, keywords, generator.gpt2_tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Train the model
    print("Starting training...")
    generator.train_model(dataloader, epochs=3)
    
    # Generate keywords for a sample text
    sample_text = texts[0]
    print("\nGenerating keywords for sample text:")
    print(f"Input text: {sample_text}")
    generated_keywords = generator.generate_keywords(sample_text)
    print(f"Generated keywords: {generated_keywords}")
    
    # Save the trained model
    print("\nSaving model...")
    generator.gpt2_model.save_pretrained('models/gpt2_lora/trained_model')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()

