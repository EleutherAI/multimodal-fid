from cleanfid.features import build_feature_extractor
from sentence_transformer import SentenceTransformer
import torch
from typing import Optional


class MultiModalFeatureExtractor(torch.nn.Module):

    def __init__(self, image_model_name: str = 'clean', text_model_path: Optional[str] = None):
        self.image_model = build_feature_extractor(image_model_name)
        self.text_model = SentenceTransformer(text_model_path)
    
    def forward(self, texts, images):
        image_features = self.image_model(images)
        text_features = self.text_model(texts)
        return torch.concat([image_features, text_features], dim=1)