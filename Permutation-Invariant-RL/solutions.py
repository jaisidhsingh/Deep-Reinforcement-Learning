import abc
import torch
import numpy as np

from solution_utils import BaselineLinearModel, PermutationInvariantPolicyNetwork


class Solution(abc.ABC):
	@abc.abstractmethod
	def clone(self):
		"""Create a deep copy of the solution to the current input/observation"""
	
	@abc.abstractmethod
	def get_action(self, x):
		"""Return the next action givent the current observation"""
	
	@abc.abstractmethod
	def get_num_features(self):
		"""Return the number of features expected by the model used for the solution"""
	
	@abc.abstractmethod
	def reset(self):
		"""Reset the current solution on rollout. Note that the weights will not be reset"""
	
	def get_parameters(self):
		params = []
		
		for p in self.policy.parameters():
			params.append(p.numpy().ravel())
		params = np.concatenate(params)
	
		return params
	
	def set_parameters(self, new_parameters):
		start, end = 0, 0
		
		for p in self.policy.parameters():
			end = start + np.prod(p.shape)
			p.data = torch.from_numpy(new_parameters[start : end].reshape(p.shape)).float()
			start = end
		
		return self
	
	def num_parameters(self):
		return len(self.get_parameters())
	

class LinearModelSolution(Solution):
	def __init__(self, config, num_features=4):
		super().__init__()
		self.kwargs = {
			'config': config,
			'num_features': num_features,
		}			

		self.datatype = torch.float32
		self.policy = BaselineLinearModel(num_features=num_features, config=config).to(self.datatype).eval()

	def clone(self):
		old_policy = self.policy
		new_solution = self.__class__(**self.kwargs)

		new_solution.laod_state_dict(
			old_policy.state_dict()
		)

		return new_solution
	
	def get_action(self, x):
		action =  self.policy(torch.from_numpy(x).to(self.datatype)).item()
		return action
	
	def get_num_features(self):
		return self.kwargs['num_features']
	
	def reset(self):
		pass


class PermutationInvariantSolution(Solution):
	def __init__(self, embedding_dim=16, project_dim=32, hidden_size=8):
		super().__init__()
		self.kwargs = {
			'embedding_dim': embedding_dim,
			'project_dim': project_dim,
			'hidden_size': hidden_size,
		}

		self.datatype = torch.float32
		self.policy = PermutationInvariantPolicyNetwork(embedding_dim, project_dim, hidden_size).to(self.datatype).eval()
		self.previous_action = 0

	def clone(self):
		old_policy = self.policy
		new_solution = self.__class__(**self.kwargs)

		new_solution.laod_state_dict(
			old_policy.state_dict()
		)

		return new_solution
	
	def get_action(self, x):
		action = self.policy(torch.from_numpy(x).to(self.datatype), self.previous_action)
		action = action.item()
		self.previous_action = action
		return action

	def reset(self):
		self.policy.sensory_neuron.hidden_state_tuple = None
		self.previous_action = 0

	def get_num_features(self):
		return None

