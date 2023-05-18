import pandas as pd
from transformers import DistilBertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
import torch
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from trainDistilBERT import DistilBertForMulticlassSequenceClassification

wandb.init(project="MarkIII_ACARIS", entity="simtoonia")



class TestACARISBERT:
	def __init__(self, mdlPath, testPath):
		self.mdlPath = mdlPath
		self.testPath = testPath
		self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
		self.model = DistilBertForMulticlassSequenceClassification.from_pretrained(self.mdlPath)

	def read_data(self, path):
		df = pd.read_csv(path, sep="|", usecols=["content", "sentiment"])
		return Dataset.from_pandas(df)

	def tokenize_data(self, dataset):
		sentMapping = {"pos": 2, "neg": 0, "neu": 1}
		tokenized = dataset.map(
			lambda x: {
				**self.tokenizer(x["content"], truncation=True, padding="max_length", max_length=512),
				"labels": torch.tensor([sentMapping[sent] for sent in x["sentiment"]])
			},
			batched=True,
			remove_columns=["content", "sentiment"]
		)
		tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
		return tokenized

	def compute_metrics(self, evalPred):
		logits, labels = evalPred
		preds = torch.argmax(torch.Tensor(logits), dim=1)
		precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
		metrics = {
			"accuracy": accuracy_score(labels, preds)
		}
		metricNames = ["precision", "recall", "f1"]
		labelNames = ["neg", "neu", "pos"]
		for metricName, metricValue in zip(metricNames, [precision, recall, f1]):
			for labelName, value in zip(labelNames, metricValue):
				metrics[f"{metricName}_{labelName}"] = float(value)
		return metrics

	def test(self):
		testDS = self.tokenize_data(self.read_data(self.testPath))

		trainingArgs = TrainingArguments(
			output_dir="./output",
			per_device_eval_batch_size=12
		)

		trainer = Trainer(
			model=self.model,
			args=trainingArgs,
			compute_metrics=self.compute_metrics,
			eval_dataset=testDS
		)
		print("Running eval ...")
		results = trainer.evaluate()
		print(f"Results:\n{results}")

if __name__ == "__main__":
	acaris_bert_test = TestACARISBERT("./output", "./datasets/test.csv")
	acaris_bert_test.test()
	wandb.finish()