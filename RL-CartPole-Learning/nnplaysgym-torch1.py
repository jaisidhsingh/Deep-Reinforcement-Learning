import numpy as np
import torchvision
import gym
import math
import matplotlib
from statistics import mean, median
import cv2
from collections import namedtuple
from matplotlib import pyplot as plt
from itertools import count

import torch
from torch import nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random
from PIL import Image

env = gym.make('CartPole-v0').unwrapped

# configure matplot
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class replayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0


	def push(self, *args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)

	def sampling(self, batchSize):
		return random.sample(self.memory, batchSize)

	def __len__(self):
		return len(self.memory)


class deepQNet(nn.Module):

	def __init__(self, height, width, outFeatures):
		super(deepQNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)

		def conv_output(size, kernel_size=5, stride=2):
			return (size - (kernel_size-1)-1)//stride +1

		conv_width = conv_output(conv_output(conv_output(width)))
		conv_height = conv_output(conv_output(conv_output(height)))

		dense_inputs = conv_width*conv_height*32
		self.head = nn.Linear(dense_inputs, outFeatures)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))


def get_cart_loc(screen_width):
	world_width = env.x_threshold*2
	scale = screen_width / world_width



def get_frame():
	screen = env.render(mode='rgb_array')#.transpose((2, 0, 1))



# arr = []
# for i in range(5):
# 	action = random.randint(0, 2)
# 	observation, reward, done, info = env.step(action)
# 	x = get_frame()
# 	arr.append(x)

# print(x)

eps_count = 100
rewards = []
frames = []

for i in range(eps_count):
	observation, done, rew, = env.reset(), False, 0
	while not done:
		action = random.randrange(0, 2)
		observation, reward, done, info = env.step(action)
		rew += reward
		x = get_frame()
		frames.append(x)


	rewards.append(rew)
print(frames)
print("average score : ", mean(rewards))

