"""
This module runs eval
"""

from sklearn.metrics import accuracy_score, f1_score
from data_loader import load_dataset
import torch



class Eval:
    def __init__(self, mdl):
        self.model = mdl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def eval(self, testDS, batchSize):
        testLoader = DataLoader(testDS, batch_size=batchSize)
        self.model.eval()

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in testLoader:
                inputIDs = batch["input_ids"].to(self.device)
                attentionMask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.mdl(inputIDs, attention_mask=attentionMask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                all_labels.extend(labels.tolist())
                all_preds.extend(preds.tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return {"accuracy": acc, "f1": f1}

if __name__ == "__main__":
    eval = Eval(mdl)
    testDS = load_dataset("test")
    metrics = eval.eval(testDS, batchSize=32)

    print(f"Metrics:\n{metrics}")