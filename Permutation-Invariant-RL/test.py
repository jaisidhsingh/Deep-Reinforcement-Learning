import torch
import numpy as np
import gym
from tasks import Task
from solutions import PermutationInvariantSolution


solution = PermutationInvariantSolution()
task = Task(render_video=True)

print(task.rollout_episode(solution))
def test():
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample() #take a random action
        print(action)
        obs, _, _, _ = env.step(action)
        print("observations: ", obs)
    env.close()

# x = torch.randn((1, 5))
# # config = {'num_layers': 3, 'layer_sizes': [64, 128, 1]}

# # model = BaselineLinearModel(x.shape[1], config)
# # 
# # print(model(x)[0])

# print(x[:, None].shape)

