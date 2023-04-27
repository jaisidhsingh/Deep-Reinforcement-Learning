import torch
import torch.nn as nn
import numpy as np
import math
import gym
from utils import *
from config import config

# utility convolutional layer with batchnorm class
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# a convolutional network to approximate the
# [Q: (state, action) --> reward] function
class DeepQNetwork(nn.Module):
    def __init__(self, config=config):
        super(DeepQNetwork, self).__init__()
        self.channels = config.in_channels
        self.height = config.image_size
        self.width = config.image_size
        self.num_actions = config.num_actions
        self.config = config

        # Conv blocks according to config
        self.blocks = []
        input_channels = self.channels       

        for block_idx in range(config.num_blocks):
            self.blocks.append(
                ConvBlock(
                    input_channels,
                    config.block_configs[block_idx]['out'],
                    kernel_size=config.block_configs[block_idx]['kernel_size'],
                    stride=config.block_configs[block_idx]['stride'],
                    padding=config.block_configs[block_idx]['pad']
                )
            )

            self.height = conv2d_output_dim(
                self.height,
                config.block_configs[block_idx]['kernel_size'],
                config.block_configs[block_idx]['stride'] 
            )

            self.width = conv2d_output_dim(
                self.width,
                config.block_configs[block_idx]['kernel_size'],
                config.block_configs[block_idx]['stride'] 
            )

            input_channels = config.block_configs[block_idx]['out']

        # Fully connected layers according to config
        self.fcs = []
        in_dim = self.width*self.height

        for layer_idx in range(config.num_layers):
            self.fcs.append(
                nn.Linear(
                    in_dim,
                    config.layer_configs[layer_idx]
                )
            )

            in_dim = config.layer_configs[layer_idx]

        self.action_head = nn.Linear(in_dim, self.num_actions)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        x = x.view(x.size(0), -1)
        for layer in self.fcs:
            x = nn.functional.relu(layer(x))

        x = self.action_head(x)
        return x


