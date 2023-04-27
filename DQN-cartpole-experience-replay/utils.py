import torch
import numpy as np
import gym
import logging
import torch.nn


def get_input():
    return None

def save_policy_checkpoint(policy, save_path):
    torch.save(policy.load_state_dict, save_path)

def load_policy_checkpoint(policy, save_path):
    torch.load(save_path, map_location=policy.load_state_dict)

def print2lines():
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("\n")

def conv2d_output_dim(dim, kernel_size, stride):
    output_dim = dim - kernel_size
    output_dim = output_dim // stride
    output_dim += 1
    return output_dim


