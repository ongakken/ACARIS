"""
This module trains.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from acaris_mdl import ACARISMdl
from acaris_ds import ACARISDs
from acaris_trainer import ACARISTrainer
from acaris_ds import ACARISDs
from acaris_collator import ACARISCollator
from user_embedder import UserEmbedder
from preprocess import Preprocessor
from data_loader import load_data
import os
import wandb
from acaris_trainer import ProgressCb
from alert import send_alert
import socket

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

config = {
	"mdl": "distilbert-base-uncased",
	"epochs": 5,
	"batchSize": 32,
	"outputDir": "./output"
}

wandb.init(project="MarkIII_ACARIS", entity="simtoonia", config=config)



class MdlTrainer:
	def __init__(self, mdl, userEmbedder):
		self.model = ACARISMdl(mdl, userEmbedder)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		wandb.watch(self.model, log="all")

	def compute_metrics(self, evalPred):
		"""
		This function computes various evaluation metrics such as accuracy, precision, recall, F1 score,
		confusion matrix, and ROC AUC score for a given set of predicted and true labels.
		
		@param evalPred `evalPred` is an object of the `EvalPrediction` class which contains the predictions
		and labels for a set of evaluation data. The `predictions` attribute contains the predicted logits
		for each example in the evaluation set, and the `label_ids` attribute contains the true labels for
		each example. The
		
		@return The function `compute_metrics` returns a dictionary containing various evaluation metrics
		such as accuracy, confusion matrix, ROC AUC score, precision, recall, and F1 score for each label
		(negative, neutral, and positive).
		"""
		logits, labels = evalPred.predictions, evalPred.label_ids
		logits = torch.softmax(torch.Tensor(logits), dim=1)
		preds = torch.argmax(logits, dim=1)
		if len(labels) > 0:
			acc = (preds == labels).sum().item() / len(labels)
		else:
			acc = 0
		precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
		accuracy = accuracy_score(labels, preds)
		confMatrix = confusion_matrix(labels, preds)
		rocAUC = roc_auc_score(labels, logits, multi_class="ovr")
		metrics = {
			"eval_accuracy": accuracy,
			"confusion_matrix": confMatrix,
			"roc_auc": rocAUC
		}
		metricNames = ["precision", "recall", "f1"]
		labelNames = ["neg", "neu", "pos"]
		for metricName, metricValue in zip(metricNames, [precision, recall, f1]):
			for labelName, value in zip(labelNames, metricValue):
				metrics[f"{metricName}_{labelName}"] = float(value)
		print(f"metrics: {metrics}")
		return metrics

	def fine_tune(self, trainLoader, valLoader, epochs, batchSize, outputDir):
		"""
		This function fine-tunes a given model using the provided training and validation data loaders, for
		a specified number of epochs and batch size, and saves the output in a specified directory.
		
		@param trainLoader A DataLoader object containing the training data.
		@param valLoader valLoader is a DataLoader object that contains the validation dataset. It is used
		during the fine-tuning process to evaluate the performance of the model on a separate set of data
		that it has not seen during training. This helps to prevent overfitting and ensures that the model
		generalizes well to new data
		@param epochs The number of times the model will iterate over the entire training dataset during
		training.
		@param batchSize The number of samples in a batch during training and evaluation.
		@param outputDir The directory where the fine-tuned model will be saved.
		"""
		# Define a callback function for reporting progress
		progressCb = ProgressCb()

		# Create output directory if it doesn't exist
		if not os.path.exists(outputDir):
			os.makedirs(outputDir)

		# Define training arguments
		trainingArgs = TrainingArguments(  
			output_dir=outputDir,
			num_train_epochs=epochs,
			per_device_train_batch_size=batchSize,
			per_device_eval_batch_size=batchSize,
			evaluation_strategy="steps",
			eval_steps=500,
			save_strategy="steps",
			save_steps=1000,
			fp16=False,
			logging_dir="./logs",
			load_best_model_at_end=True,
			metric_for_best_model="eval_accuracy"
		)

		trainer = ACARISTrainer(
			model=self.model,
			args=trainingArgs,
			trainLoader=trainLoader,
			evalLoader=valLoader,
			compute_metrics=self.compute_metrics,
			callbacks=[ProgressCb()]
		)
		try:
			send_alert("**WE ARE EVALUATING**", f"Eval started on `{socket.gethostname()}`", "low", 5000, sound=False, discord=True)
			trainer.evaluate()
			send_alert("**EVALUATION COMPLETE**", "Eval complete", "low", 5000, sound=False, discord=True)
		except Exception as e:
			send_alert("!! EVAL FAILED !!", str(e), "critical", 50000, sound=True, discord=True)
			raise
		try:
			send_alert("**WE ARE TRAINING**", f"Training started on `{socket.gethostname()}`", "low", 5000, sound=False, discord=True)
			trainer.train()
			send_alert("**TRAINING COMPLETE**", "Training complete", "low", 5000, sound=False, discord=True)
		except Exception as e:
			send_alert("!! TRAINING FAILED !!", str(e), "critical", 50000, sound=True, discord=True)
			raise

if __name__ == "__main__":
	mdl = config["mdl"]
	userEmbedder = UserEmbedder(userEmbeddingSize=11)
	loop = MdlTrainer(mdl=mdl, userEmbedder=userEmbedder)
	preprocessor = Preprocessor(mdl)
	collator = ACARISCollator(loop.device, userEmbedder)

	batchSize = config["batchSize"]

	train = load_data("train")
	print(f"Train len: {len(train)}")
	val = load_data("val")

	trainDS = ACARISDs(train, preprocessor, userEmbedder)
	print(f"train sent counts: {train['sentiment'].value_counts()}")
	valDS = ACARISDs(val, preprocessor, userEmbedder)
	print(f"val sent counts: {val['sentiment'].value_counts()}")
	print(f"trainDS len: {len(trainDS)}")
	trainLoader = DataLoader(trainDS, batch_size=config["batchSize"], shuffle=True, collate_fn=collator, drop_last=False)
	valLoader = DataLoader(valDS, batch_size=config["batchSize"], shuffle=False, collate_fn=collator, drop_last=False)

	loop.fine_tune(trainLoader=trainLoader, valLoader=valLoader, epochs=config["epochs"], batchSize=config["batchSize"], outputDir=config["outputDir"])

	wandb.finish()
