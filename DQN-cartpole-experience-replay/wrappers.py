from itertools import count
from collections import deque, namedtuple 
from wrapper import *
from models.dqn import DeepQNetwork
from utils import *
import math
import torch
import random
import gym
from config import config

# a class wrapping our training methodology
class Trainer():
    def __init__(self, training_strategy, config):
        self.config = config
        self.plot_eps
        self.env = gym.make(config.env_name)
        self.env._max_episode_steps = config.max_steps
        self.visualize_training = config.render_video
        self.strategy = training_strategy

    def train(self, agent, experience):
        optimizer = torch.optim.RMSprop(agent.policy_model.parameters())

        steps_done = 0
        for eps_idx in range(self.config.num_eps):
            self.env.reset()
            previous_input = get_input()
            current_input = get_input()
            state = current_input - previous_input

            for idx in count():
                action = self.strategy.explore_vs_exploit(agent.policy, state, steps_done)
                new_observation, reward, done, info = self.env.step(action)
                reward = torch.tensor(reward).unsqueeze(0).to(self.config.device)

                # new state
                previous_input = current_input
                current_input = get_input()
                if not done:
                    next_state = current_input - previous_input
                else:
                    next_state = None

                # store experience
                experience.add_to_buffer(state, action, next_state, reward)
                state = next_state

                agent.optimize_models(experience, optimizer, config)

                # break if game over
                if done:
                    break
            
            # update target model params after a new eps
            # which is also a hyperparameter to tune
            if eps_idx % config.hyperparams.target_update == 0:
                agent.target_model.load_state_dict(
                    agent.policy_model.state_dict()
                )
        
        print(f"Trained for {config.num_eps} episode(s)")

# a class to wrap the testing of our policy network
# on the OpenAI Gym environment 'CartPole-v0' for us
class Tester():
    def __init__(self, config):
        self.env = gym.make(config.env_name)
        self.max_steps = config.max_steps
        self.env._max_episode_steps = self.max_steps
        self.visualize = config.render_video

    def reset_env(self):
        # reset environment
        return self.env.reset()

    def get_single_ep_reward(self, policy):
        # # reset policy before run
        # policy.reset()

        # initialize a random observation
        # by resetting the environment
        obs = self.reset_env()

        # optional test video rendering
        if self.visualize:
            self.env.render()
        
        current_ep_reward = 0
        # represents whether game is over or not
        done = False

        while not done:
            action = policy.get_next_move(obs)
            action = int(action)
            new_obs, reward, done, _ = self.env.step(action)
            obs = new_obs
            current_ep_reward += reward

        # return acculumated reward         
        return current_ep_reward
    
    def get_all_ep_rewards(self, policy, num_ep):
        # return a list of rewards for
        # some number of episodes 
        return [self.get_single_ep_reward(policy) for _ in range(num_ep)]


# a class wrapping our agent which optimizes itself
# from the experience collected in the replay buffer
# using the Epsilon Greedy Strategy to explore/exploit
class AgentWithExperience():
    def __init__(self, name, policy_model, target_model):
        self.name = name
        self.policy_model = policy_model
        self.target_model = target_model
        self.transition = namedtuple('transition', 
            (
                'current_state',
                'action',
                'reward',
                'next_state'
            )
        ) 

    def get_next_move(self, state):
        # choose the action according to our
        # policy with the highest q_value
        next_move = self.policy_model(state).max(1)[1].view(-1, 1)
        return next_move

    def optimize_models(self, memory, optimizer, config):
        # return None if we do not have a
        # single batch of memory to replay
        if len(memory) < config.hyperparams.batch_size:
            return None

        # initialize experience replay        
        transitions = memory.sample_random_batch()
        batch = self.transition(*zip(transitions))

        # find indices of intermediate states
        # also called a mask
        mask = tuple(map(lambda state: state is not None, batch.next_state))
        mask_tensor = torch.tensor(mask)

        # find next states which do not 
        # occur after the game is done
        intermediate_next_states = torch.cat(
            [state for state in batch.next_state if state is not None]
        )

        # concatenate into tensors of shape:
        # [Batch, ...] so that we can them
        # as inputs to our model        
        batched_states = torch.cat(batch.state)
        batched_actions = torch.cat(batch.action)
        batched_rewards = torch.cat(batch.reward)

        # get q_values for each action from our model
        q_values = self.policy_model(batched_states).gather(1, batched_actions)

        tmp_nexts = torch.zeros(config.hyperparams.batch_size, device=config.device)
        # update next states using target model output
        # with input as intermediate states made above
        tmp_nexts[mask_tensor] = self.target_model(intermediate_next_states).max(1)[0].detach()

        # calculate for loss
        exp_q_values = tmp_nexts*config.hyperparams.gamma + batched_rewards

        # Huber Loss for Q-Learnign
        loss_fn = torch.nn.SmoothL1Loss()
        loss = loss_fn(q_values, exp_q_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        # clamp model weights between -1 to 1
        # after backprop
        for p in self.policy_model.parameters():
            p.grad.clamp_(-1, 1)
        optimizer.step()


# a class wrapping the strategy behind 
# what action to take at which state
class EpsilonGreedyStrategy():
    def __init__(self, config):
        # class wrapping the action selection
        # using hyperparameters defined in the config
        self.epsilon_start = config.hyperparams.epsilon_start 
        self.epsilon_end = config.hyperparams.epsilon_end
        self.epsilon_decay = config.hyperparams.epsilon_decay
        self.device = config.device

    def explore_vs_exploit(self, policy, state, steps_done):
        # computing epsilon threshold to conditionally 
        # select the next move to make according to our
        # Epsilon Greedy Strategy
        random_sample = random.random()
        epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1.*steps_done / self.epsilon_decay)

        steps_done += 1

        # condition for exploitaion of the environment
        # that is, to utilize learned knowledge for next move
        if random_sample > epsilon_threshold:
            with torch.no_grad():
                next_move = policy.get_next_move(state)
                return next_move, 'exploited' # to visualize and infer from later

        # if the threshold is greater then select
        # a random action to explore the environment 
        else:
            return torch.tensor([[
                random.randrange(
                    policy.policy_model.num_actions
                )
            ]], 
                device=self.device, 
                dtype=torch.long
            ), 'exploited' # to visualize and infer from later


# a class wrapping our replay buffer
# to store and sample experience from
class ReplayBuffer():
    def __init__(self, config):
        self.max_buffer_size = config.max_buffer_size
        self.batch_size = config.batch_size
        self.buffer = deque([], maxlen=self.max_buffer_size)
        self.transition = namedtuple('transition', 
            (
                'current_state',
                'action',
                'reward',
                'next_state'
            )
        )
    
    def sample_random_batch(self):
        return random.sample(self.buffer, self.batch_size)

    def add_to_buffer(self, *args):
        self.buffer.append(
            self.transition(*args)
        )

    def rm_latest_memory(self):
        self.buffer.pop()

    def __len__(self):
        return len(self.buffer)

