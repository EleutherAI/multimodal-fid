import torch
from transformers import AutoModel


MODEL_PATH = 'sentence-transformers/all-roberta-large-v1'


def mean_pooling(model_output):
    token_embeddings = model_output[0]
    return torch.mean(token_embeddings, 1)


class SentenceTransformer(torch.nn.Module):

    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path if model_path else MODEL_PATH
        self.transformer = AutoModel.from_pretrained(model_path)
    
    def forward(self, batch):
        return mean_pooling(self.transformer(batch))
