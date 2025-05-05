"""
Both Manual and convolutional patching is implemented in this file.
"""

import torch
import torch.nn as nn

class ConvolutionalPatchEmbedding(nn.Module):
    def __init__(self, embedding_dimension, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(embedding_dimension, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)

    def forward(self, x):
        return self.flatten(self.conv(x))