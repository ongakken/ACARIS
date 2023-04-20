"""
The preprocess module contains functions for preprocessing text data.
"""

from transformers import AutoTokenizer


class Preprocessor:
    def __init__(self, mdl):
        self.tokenizer = AutoTokenizer.from_pretrained(mdl)

    def tokenize(self, text, maxLen, padding=True, truncation=True, returnTensors="pt"):
        tokens = self.tokenizer(text, max_length=max_length, padding=padding, truncation=truncation, return_tensors=return_tensors)
        #tokens["input_ids"] = torch.cat([userEmbedding, tokens["input_ids"]], dim=-1)
        return tokens

if __name__ == "__main__":
    mdl = "distilbert-base-uncased"
    preprocessor = Preprocessor(mdl)
    text = "Fuck me, hard!"
    tokens = preprocessor.tokenize(text, 64)
    print(f"Sweet, sweet tokens:\n{tokens}")