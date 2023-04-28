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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



class MdlTrainer:
    def __init__(self, mdl, userEmbedder):
        self.model = ACARISMdl(mdl, userEmbedder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fine_tune(self, trainLoader, valLoader, epochs, batchSize, outputDir):
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        trainingArgs = TrainingArguments(  
            output_dir=outputDir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batchSize,
            per_device_eval_batch_size=batchSize,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )

        trainer = ACARISTrainer(
            model=self.model,
            args=trainingArgs,
            trainLoader=trainLoader,
            evalLoader=valLoader,
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
    userEmbedder = UserEmbedder()
    trainer = MdlTrainer(mdl, userEmbedder)
    preprocessor = Preprocessor(mdl)
    collator = ACARISCollator()

    batchSize = 32

    train = load_data("train")
    val = load_data("val")

    trainDS = ACARISDs(train, preprocessor, userEmbedder)
    valDS = ACARISDs(val, preprocessor, userEmbedder)

    trainLoader = DataLoader(trainDS, batch_size=batchSize, shuffle=True, collate_fn=collator)
    valLoader = DataLoader(valDS, batch_size=batchSize, shuffle=False, collate_fn=collator)

    trainer.fine_tune(trainLoader=trainLoader, valLoader=valLoader, epochs=5, batchSize=32, outputDir="./output")