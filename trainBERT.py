"""
This mod fine-tunes a BERT model on the ACARIS dataset for comparison with ACARISMdl.
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer, AdamW, EarlyStoppingCallback, PreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import huggingface_hub
import os

config = {
	"mdl": "bert-base-uncased",
	"epochs": 5,
	"batchSize": 14,
    "maxLen": 512,
    "warmupSteps": 0.1, # proportion of total steps, NOT absolute
    "weightDecay": 0.02,
	"outputDir": "./output",
    "earlyStopping": True,
    "earlyStoppingPatience": 2,
    "initlr": 5e-5,
    "epsilon": 1e-8
}

wandb.init(project="MarkIII_ACARIS", entity="simtoonia", config=config)



class ACARISBERT:
    def __init__(self, trainPath, valPath):
        self.trainPath = trainPath
        self.valPath = valPath
        self.tokenizer = BertTokenizerFast.from_pretrained(config["mdl"])
        self.model = BertForSequenceClassification.from_pretrained(config["mdl"], num_labels=3, id2label={0: "neg", 1: "neu", 2: "pos"}, label2id={"neg": 0, "neu": 1, "pos": 2})



    def read_data(self, path):
        df = pd.read_csv(path, sep="|", usecols=["content", "sentiment"])
        return Dataset.from_pandas(df)
    
    def tokenize_data(self, dataset):
        sentMapping = {"pos": 2, "neg": 0, "neu": 1}
        tokenized = dataset.map(
            lambda x: {
                **self.tokenizer(x["content"], truncation=True, padding="max_length", max_length=config["maxLen"]),
                "labels": torch.tensor([sentMapping[sent] for sent in x["sentiment"]])
            },
            batched=True,
            remove_columns=["content", "sentiment"]
        )
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized
    
    def get_data_loaders(self, trainDS, valDS):
        trainLoader = DataLoader(trainDS, batch_size=config["batchSize"], shuffle=False)
        valLoader = DataLoader(valDS, batch_size=config["batchSize"], shuffle=False)
        return trainLoader, valLoader
    
    def compute_metrics(self, evalPred):
        logits, labels = evalPred
        preds = torch.argmax(torch.Tensor(logits), dim=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
        metrics = {
            "accuracy": accuracy_score(labels, preds),
        }
        metricNames = ["precision", "recall", "f1"]
        labelNames = ["neg", "neu", "pos"]
        for metricName, metricValue in zip(metricNames, [precision, recall, f1]):
            for labelName, value in zip(labelNames, metricValue):
                metrics[f"{metricName}_{labelName}"] = float(value)
        return metrics
    
    def train(self):
        trainDS = self.tokenize_data(self.read_data(self.trainPath))
        valDS = self.tokenize_data(self.read_data(self.valPath))

        totalSteps = len(trainDS) // config["batchSize"] * config["epochs"]
        warmupSteps = int(totalSteps * config["warmupSteps"])
        
        trainingArgs = TrainingArguments(
            output_dir=config["outputDir"],
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batchSize"],
            per_device_eval_batch_size=config["batchSize"],
            warmup_steps=warmupSteps,
            weight_decay=config["weightDecay"],
            logging_dir="./logs",
            logging_steps=100,
            learning_rate=config["initlr"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=5,
            adam_epsilon=config["epsilon"],
            report_to="wandb",
            fp16=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=trainingArgs,
            train_dataset=trainDS,
            eval_dataset=valDS,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config["earlyStoppingPatience"])]
        )
        print(f"Number of parameters: {trainer.model.num_parameters()}")
        print("Running eval ...")
        trainer.evaluate()
        print("Running training ...")
        trainer.train()
        print("Saving model ...")
        trainer.save_model(config["outputDir"])
        
        
if __name__ == "__main__":
    acaris_bert = ACARISBERT("./datasets/train.csv", "./datasets/val.csv")
    acaris_bert.train()
    wandb.finish()
