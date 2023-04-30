"""
This mod handles user embeddings.
"""

import torch
from feature_extract import FeatExtractor


class UserEmbedder:
    def __init__(self, userEmbeddingSize=10):
        self.mdl = "distilbert-base-uncased"
        self.userEmbeddings = {}
        self.userEmbeddingSize = userEmbeddingSize
        self.extractor = FeatExtractor(self.mdl)
        self.msgs = self.extractor.read_msgs_from_file("./datasets/sents_merged_cleaned.csv")

    def get_user_embedding(self, userID):
        if userID not in self.userEmbeddings:
            feats = self.extractor.extract_and_store_feats(self.msgs, userID)
            if feats is not None:
                self.userEmbeddings[userID] = self.create_user_embedding(feats)
            else:
                #self.userEmbeddings[userID] = torch.zeros(13, dtype=torch.float)
                self.userEmbeddings[userID] = None
        return self.userEmbeddings[userID]

    def update_user_embedding(self, userID):
        feats = self.extractor.extract_and_store_feats(self.msgs, userID)
        self.userEmbeddings[userID] = self.create_user_embedding(feats)

    def create_user_embedding(self, feats):
        if not feats:
            return None

        for k, v in feats.items():
            if v is None:
                print(f"None value for {k}")
                raise ValueError
        print(f"\nfeats: {feats} of len {len(feats)}\n")
        feats = [v for v in feats.values() if isinstance(v, (int,float))]
        print(f"\nfeats: {feats} of len {len(feats)}\n")
        emb = torch.tensor(feats, dtype=torch.float)
        return emb