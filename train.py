"""
This module trains.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments


class MdlTrainer:
    def __init__(self, mdl):
        self.model = AutoModelForSequenceClassification.from_pretrained(mdl)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fine_tune(self, trainDS, valDS, epochs, batchSize, outputDir):
        trainingArgs = TrainingArguments(
            output_dir=outputDir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batchSize,
            per_device_eval_batch_size=batchSize,
            evaluation_strategy="epoch",
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )

        trainer = MdlTrainer(
            model=self.model,
            args=trainingArgs,
            train_dataset=trainDS,
            eval_dataset=valDS,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def compute_metrics(self, evalPred):
        logits, labels = evalPred
        predictions = torch.argmax(logits, dim=-1)
        
        acc = (predictions == labels).sum().item() / len(labels)
        return {"accuracy": acc}

if __name__ == "__main__":
    mdl = "distilbert-base-uncased"
    trainer = MdlTrainer(mdl)

    trainDS = load_dataset("train")
    valDS = load_dataset("val")

    trainer.fine_tune(trainDS, valDS, epochs=5, batchSize=32, outputDir="./output")