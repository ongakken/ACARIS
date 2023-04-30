import torch
import torch.nn as nn
from transformers import AutoModel
from user_embedder import UserEmbedder


class ACARISMdl(nn.Module):
	def __init__(self, mdl, userEmbedder, numLabels = 3):
		super().__init__()
		self.bert = AutoModel.from_pretrained(mdl)
		self.classifier = nn.Linear(self.bert.config.hidden_size + 13, numLabels)
		self.userEmbedder = userEmbedder

	def forward(self, input_ids, attention_mask, userEmbs, labels=None):
		bertOut = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		tokenEmbs = bertOut.last_hidden_state[:, 0, :]
		userEmbs = userEmbs.to(tokenEmbs.device)
		print(f"tokenEmbs.shape: {tokenEmbs.shape}")
		print(f"userEmbs.shape: {userEmbs.shape}")
		combinedEmbs = torch.cat((tokenEmbs, userEmbs), dim=-1)
		logits = self.classifier(combinedEmbs)

		return {"logits": logits}