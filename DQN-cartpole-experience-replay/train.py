from wrappers import *
from utils import *
from dqn import DeepQNetwork
from config import config


# Initialize the strategy 
# for exploration and exploitation
strategy = EpsilonGreedyStrategy()

# Initialize policy networks
policy = DeepQNetwork()
target = DeepQNetwork()
target.load_state_dict(
    policy.state_dict()
)

# Initialize agent
agent = AgentWithExperience(
    name='dqn',
    policy_model=policy,
    target_model=target
)

# Initialize replay buffer
# to store experience
experience = ReplayBuffer()

# Train the agent
trainer = Trainer(strategy, config)
trainer.train(agent, experience)