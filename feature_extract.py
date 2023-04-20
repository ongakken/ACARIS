"""
This module extracts features from tokenized input sequences.
"""

import torch
from transformers import AutoModel


class FeatExtractor:
    def __init__(self, mdl):
        self.model = AutoModel.from_pretrained(mdl)

    def extract_feats(self, tokens):
        with torch.no_grad():
            outputs = self.model(**tokens)
            feats = outputs.last_hidden_state
        return feats

if __name__ == "__main__":
    mdl = "distilbert-base-uncased"
    extractor = FeatExtractor(mdl)

    tokens = preprocessor.tokenize("Fuck me slowly!")

    feats = extractor.extract_feats(tokens)
    print(f"Feats:\n{feats}")