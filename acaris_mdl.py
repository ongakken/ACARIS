import torch
import torch.nn as nn
from transformers import AutoModel
from user_embedder import UserEmbedder


class ACARISMdl(nn.Module):
	def __init__(self, mdl, userEmbedder):
		super().__init__()
		self.bert = AutoModel.from_pretrained(mdl)
		self.classifier = nn.Linear(self.bert.config.hidden_size + userEmbedder.userEmbeddingSize, self.bert.config.num_labels)
		self.userEmbedder = userEmbedder

	def forward(self, input_ids, attention_mask, userIDs):
		userEmbeddings = torch.stack([self.userEmbedder.get_user_embedding(userID.item()) for userID in userIDs]).to(input_ids.device)
		bertOut = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		tokenEmbs = bertOut.last_hidden_state[:, 0, :]
		combinedEmbs = torch.cat((tokenEmbs, userEmbeddings), dim=-1)
		logits = self.classifier(combinedEmbs)
		return logits