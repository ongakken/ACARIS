"""
This module sets up a Discord bot called "ACARIS".
"""

import discord
from discord.ext import commands

from preprocess import Preprocessor
from feature_extract import FeatExtractor
from train import Trainer
from eval import Eval


class Bot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def on_ready(self):
        print(f"Init'ed and conn'd to Discord as {self.user}")

    async def on_message(self, message):
        if message.author == self.user:
            return

if __name__ == "__main__":
    bot = Bot(command_prefix="\\")
    bot.run("")