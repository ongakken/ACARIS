"""
The message buffer is a queue of messages that are waiting to be prepped for training.
"""

import discord
from feature_extract import FeatExtractor


class MsgBuffer:
	def __init (self, bufferSize, extractor):
		self.bufferSize = bufferSize
		self.buffer = []
		self.extractor = extractor

	def add_msg (self, msg):
		self.buffer.append(msg)
		
		if len(self.buffer) > self.bufferSize:
			self.process_buffer()

	def process_buffer (self):
		msgDicts = [msg.to_dict() for msg in self.buffer]
		feats = self.extractor.extract_feats(msgDicts)
		print(f"Feats:\n{feats}")

		# TODO: Send feats to Trainer

		self.buffer.clear()