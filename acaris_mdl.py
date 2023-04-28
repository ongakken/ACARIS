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

	def forward(self, input_ids, attention_mask, userIDs, labels=None):
		#userIndices = [self.userEmbedder.get_user_index(userID.item()) for userID in userIDs]
		userIndices = [self.userEmbedder.get_user_index(userID) for userID in userIDs]
		userIndices = torch.tensor(userIndices, dtype=torch.long, device=input_ids.device)
		userEmbeddings = [self.userEmbedder.get_user_embedding(userID) for userID in userIDs]
		userEmbeddings = torch.stack(userEmbeddings)
		bertOut = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		tokenEmbs = bertOut.last_hidden_state[:, 0, :]
		userEmbeddings = userEmbeddings.to(tokenEmbs.device)
		combinedEmbs = torch.cat((tokenEmbs, userEmbeddings), dim=-1)
		logits = self.classifier(combinedEmbs)

		if labels is not None:
			lossFct = nn.CrossEntropyLoss()
			loss = lossFct(logits.view(-1, self.bert.config.num_labels), labels.view(-1))
			return {"loss": loss, "logits": logits}
		else:
			return {"logits": logits}