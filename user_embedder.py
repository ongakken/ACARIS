"""
This mod handles user embeddings.
"""

import torch


class UserEmbedder:
    def __init__(self, userEmbeddingSize=10):
        self.userIndices = {}
        self.userEmbeddings = {}
        self.userEmbeddingSize = userEmbeddingSize
        self.currentIndex = 0

    def get_user_index(self, userID):
        if userID not in self.userIndices:
            self.userIndices[userID] = self.currentIndex
            self.currentIndex += 1
        return self.userIndices[userID]

    def get_user_embedding(self, userID):
        if userID not in self.userEmbeddings:
            self.userEmbeddings[userID] = torch.rand(self.userEmbeddingSize)
        return self.userEmbeddings[userID]

    def update_user_embedding(self, userID, newEmbedding):
        self.userEmbeddings[userID] = newEmbedding