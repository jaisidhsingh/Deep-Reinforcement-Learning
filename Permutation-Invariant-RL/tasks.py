import gym
import gym_cartpole_swingup
import numpy as np


ACTUAL_FEATURES_DIM = 4

class IncompatibleFeatureDimensions(Exception):
   """
   Raised when the number of features in the observation passed 
   are incompatible with the number of features in the observation
   for our CartPole task
   """


class Task():
	def __init__(self, render_video=False, permute_before_rollout=False, noise_num_features=0, env_seed=None, permutation_noise_seed=None, max_steps=10000):
		self.render_video = render_video
		self.permute_before_rollout = permute_before_rollout
		self.noise_num_features = noise_num_features
		self.env_seed = env_seed
		self.permutation_noise_seed = permutation_noise_seed
		self.max_steps = max_steps

		self.env = gym.make('CartPole-v0')
		self.env._max_episode_steps = self.max_steps

		self.num_features = self.noise_num_features + ACTUAL_FEATURES_DIM

		self.indices_to_permute = np.arange(self.num_features)
		self.noise_std = 0.1

		self.env.seed(self.env_seed)
		self.random_gen = np.random.RandomState(seed=permutation_noise_seed)

	def prepare_and_rollout(self):
		self.indices_to_permute = np.arange(self.num_features)

		if self.permute_before_rollout:
			self.random_gen.shuffle(self.indices_to_permute)
	
	def add_noise_and_permute(self, x):
		noise = self.random_gen.randn(self.noise_num_features) * self.noise_std
		noisy_observation = np.concatenate([x, noise], axis=0)
		noisy_and_permuted = noisy_observation[self.indices_to_permute]

		return noisy_and_permuted
	
	def reset(self):
		self.env.reset()
	
	def rollout_episode(self, solution):
		self.prepare_and_rollout()
		solution.reset()

		current_observation = self.env.reset()

		if self.render_video:
			self.env.render()
		
		reward_for_current_episode = 0
		done = False

		while not done:
			new_observation = self.add_noise_and_permute(current_observation)
			action = int(solution.get_action(new_observation))
			observation, reward, done, info = self.env.step(action)
			current_observation = observation
			reward_for_current_episode += reward

			if self.render_video:
				self.env.render()
		
		return reward_for_current_episode
	
	def rollout_episodes(self, solution, num_episodes):
		return [self.rollout_episode(solution) for i in range(num_episodes)]
