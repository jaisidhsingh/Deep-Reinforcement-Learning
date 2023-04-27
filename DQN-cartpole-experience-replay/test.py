from wrappers import *
from config import config
from dqn import DeepQNetwork
import argparse


# Add and parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--num-eps",
    type=int,
    required=True
)

parser.add_argument(
        "--policy-checkpoint",
        type=str,
        required=False
)

parser.add_argument(
    "--target-checkpoint",
    type=str,
    required=False
)

parser.add_argument(
    "--render",
    type=bool,
    required=False
)

args = parser.parse_args()

# set render video value in config to argument value
config.render_video = args.render

# Initialize policy and target networks
policy = DeepQNetwork()
target = DeepQNetwork()

# Load parameters from checkpoint
policy.load_state_dict(
    torch.load(args.policy_checkpoint, map_location=config.device)
).to(config.device)
target.load_state_dict(
    torch.load(args.target_checkpoint, map_location=config.device)
).to(config.device)

# Create agent
agent = AgentWithExperience(
    name='dqn',
    policy_model=policy,
    target_model=target
)

# Test policy in our environment
tester = Tester(config)
tester.get_all_ep_rewards(agent, args.num_eps)
