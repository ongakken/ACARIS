from transformers import DistilBertTokenizerFast, pipeline
import torch
import sys
from trainDistilBERT import DistilBertForMulticlassSequenceClassification
import os
import numpy as np
import wandb
import shap

wandb.init(mode="disabled")
os.environ["WANDB_MODE"] = "disabled"

class InferACARISBERT:
	def __init__(self, mdlPath):
		self.mdlPath = mdlPath
		self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.mdlPath)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = DistilBertForMulticlassSequenceClassification.from_pretrained(self.mdlPath).to(self.device)

		# TODO: #10 define a custom Transformers pipeline
		self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, top_k=None, device=0)

	def prepare_input(self, text):
		inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt").to(self.device)
		return inputs

	def predict(self, inputs):
		outputs = self.model(**inputs)
		probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
		return probs.detach().cpu().numpy()

	def infer(self, text):
		if text:
			inputs = self.prepare_input(text)
			pred = self.predict(inputs)
			print(f"Input: {text}\nPrediction: {pred}\n")
			maxProb = np.argmax(pred, axis=-1)
			label = self.model.config.id2label[maxProb[0]]
			confidence = pred[0][maxProb[0]]
			print(f"Label: {label}\nConfidence: {confidence}\n")
		else:
			for text in sys.stdin:
				text = text.strip()
				inputs = self.prepare_input(text)
				pred = self.predict(inputs)
				print(f"Input: {text}\nPrediction: {pred}\n")
				maxProb = np.argmax(pred, axis=-1)
				label = self.model.config.id2label[maxProb[0]]
				confidence = pred[0][maxProb[0]]
				print(f"Label: {label}\nConfidence: {confidence}\n")

	def explain(self, text):
		labels = sorted(self.model.config.label2id, key=self.model.config.label2id.get)
		explainer = shap.Explainer(self.classifier)
		shap_values = explainer(text)

		print(self.classifier(text))

		# shap.initjs()
		shap.plots.text(shap_values)
		shap.plots.bar(shap_values.mean(0))


if __name__ == "__main__":
	acaris_bert_infer = InferACARISBERT("ongknsro/ACARIS_BASELINE-DistilBERT-evalSpecimen1-batchSize32")
	#acaris_bert_infer.infer("I am a student.")
	acaris_bert_infer.explain(" I love her! <3")