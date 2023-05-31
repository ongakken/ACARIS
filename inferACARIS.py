import torch
from transformers import DistilBertTokenizerFast
from user_embedder import UserEmbedder
from trainACARIS import DistilBertForMulticlassSequenceClassification


class InferACARIS:
	def __init__(self, mdlPath):
		self.model = DistilBertForMulticlassSequenceClassification.from_pretrained(mdlPath)
		self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
		self.userEmbedder = UserEmbedder(userEmbeddingSize=11)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)

	def predict(self, uids, contents):
		userEmbs = [self.userEmbedder.get_user_embedding(uid) for uid in uids]
		inputs = self.tokenizer(contents, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
		inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
		userEmbs = torch.stack(userEmbs).to(self.device)

		self.model.eval()
		with torch.no_grad():
			outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], userEmbs=userEmbs)

		probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
		preds = outputs.logits.argmax(dim=1).cpu().numpy()

		id2label = {0: "neg", 1: "neu", 2: "pos"}
		labels = [id2label[pred] for pred in preds]

		return labels, probs


if __name__ == "__main__":
	infer = InferACARIS("./output/checkpoint-3540/")
	uids = ["Reknez#9257"]
	contents = ["ok"]
	labels, probs = infer.predict(uids, contents)
	print(labels, probs)