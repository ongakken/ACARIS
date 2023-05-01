"""
This module sets up a Discord bot called "ACARIS".
"""

import discord
from discord.ext import commands

from preprocess import Preprocessor
from feature_extract import FeatExtractor
from train import Trainer
from eval import Eval
from user_embedder import UserEmbedder
from msg_buffer import MsgBuffer

from dotenv import load_dotenv
import os


# This is a Python class for a Discord bot that uses natural language processing to extract features
# from messages and train/evaluate a model.
class Bot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mdl = "distilbert-base-uncased"
        self.preprocessor = Preprocessor(self.mdl)
        self.extractor = FeatExtractor(self.mdl)
        self.trainer = Trainer(self.mdl)
        self.eval = Eval(self.mdl)
        self.userEmbedder = UserEmbedder()

        bufferSize = 250
        msgBuffer = MsgBuffer(bufferSize, extractor)


    async def on_ready(self):
        print(f"Init'ed and conn'd to Discord as {self.user}")

    async def on_message(self, message):
        if message.author == self.user:
            return

        #userID = message.author.id
        #userEmbedding = self.userEmbedder.get_user_embedding(userID)

        # TODO: Train and eval here.

if __name__ == "__main__":
    bot = Bot(command_prefix="\\")
    load_dotenv()
    token = os.getenv("DSC")
    bot.run(token)