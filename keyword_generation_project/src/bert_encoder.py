from transformers import BertModel
import torch

class BertEncoder:
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def get_embeddings(self, text_batch):
        with torch.no_grad():
            outputs = self.model(**text_batch)
            embeddings = outputs.last_hidden_state
        return embeddings