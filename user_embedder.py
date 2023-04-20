"""
This mod handles user embeddings.
"""

import torch


class UserEmbedder:
    def __init__(self, userEmbeddingSize):
        self.userEmbeddings = {}
        self.userEmbeddingSize = userEmbeddingSize

    def get_user_embedding(self, userID):
        if userID not in self.userEmbeddings:
            self.userEmbeddings[userID] = torch.rand(self.userEmbeddingSize)
        return self.userEmbeddings[userID]

    def update_user_embedding(self, userID, newEmbedding):
        self.userEmbeddings[userID] = newEmbedding