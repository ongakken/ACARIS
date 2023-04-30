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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

config = {
    "mdl": "distilbert-base-uncased",
    "epochs": 1,
    "batchSize": 8,
    "outputDir": "./output"
}

wandb.init(project="MarkIII_ACARIS", entity="simtoonia", config=config)



class MdlTrainer:
    def __init__(self, mdl, userEmbedder):
        self.model = ACARISMdl(mdl, userEmbedder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu") # TODO: #1 fuck me, not enough VRAM. fuck me hard. ffs
        self.model.to(self.device)
        wandb.watch(self.model, log="all")

    def fine_tune(self, trainLoader, valLoader, epochs, batchSize, outputDir):
        progressCb = ProgressCb()
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        trainingArgs = TrainingArguments(  
            output_dir=outputDir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batchSize,
            per_device_eval_batch_size=batchSize,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            fp16=True,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )

        trainer = ACARISTrainer(
            model=self.model,
            args=trainingArgs,
            trainLoader=trainLoader,
            evalLoader=valLoader,
            compute_metrics=self.compute_metrics,
            progressCb=progressCb
        )

        trainer.train()

    def compute_metrics(self, evalPred):
        logits, labels = evalPred
        predictions = torch.argmax(logits, dim=-1)
        
        acc = (predictions == labels).sum().item() / len(labels)
        wandb.log({"accuracy": acc})
        return {"accuracy": acc}

if __name__ == "__main__":
    mdl = config["mdl"]
    userEmbedder = UserEmbedder()
    trainer = MdlTrainer(mdl=mdl, userEmbedder=userEmbedder)
    preprocessor = Preprocessor(mdl)
    collator = ACARISCollator()

    batchSize = config["batchSize"]

    train = load_data("train")
    print(f"Train len: {len(train)}")
    val = load_data("val")

    trainDS = ACARISDs(train, preprocessor, userEmbedder)
    valDS = ACARISDs(val, preprocessor, userEmbedder)

    trainLoader = DataLoader(trainDS, batch_size=config["batchSize"], shuffle=True, collate_fn=collator, drop_last=False)
    valLoader = DataLoader(valDS, batch_size=config["batchSize"], shuffle=False, collate_fn=collator, drop_last=False)

    trainer.fine_tune(trainLoader=trainLoader, valLoader=valLoader, epochs=config["epochs"], batchSize=config["batchSize"], outputDir=config["outputDir"])

    wandb.finish()