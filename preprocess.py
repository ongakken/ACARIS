"""
The preprocess module contains methods for preprocessing text data.
"""

from transformers import AutoTokenizer
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



class Preprocessor:
    def __init__(self, mdl):
        self.tokenizer = AutoTokenizer.from_pretrained(mdl)

    def tokenize(self, text, maxLen=512, padding=True, truncation=True, returnTensors="pt"):
        if type(text) == float:
            raise ValueError(f"Got float: {text}")
        lenTokenized = len(self.tokenizer.tokenize(text))
        if lenTokenized > maxLen:
            with open("longerThanMaxLen.txt", "a") as f:
                f.write(f"Long seq:\n\n{text}\n\n\n\n")
        tokens = self.tokenizer(text, max_length=maxLen, padding=padding, truncation=truncation, return_tensors=returnTensors)
        #tokens["input_ids"] = torch.cat([userEmbedding, tokens["input_ids"]], dim=-1)
        #tokens["userID"] = torch.tensor(userID, dtype=torch.long)
        return tokens

if __name__ == "__main__":
    mdl = "distilbert-base-uncased"
    preprocessor = Preprocessor(mdl)
    text = "F* me, hard!"
    tokens = preprocessor.tokenize(text)
    print(f"Sweet, sweet tokens:\n{tokens}")
    print(f"Decoded tokens:\n{preprocessor.tokenizer.decode(tokens['input_ids'][0])}")