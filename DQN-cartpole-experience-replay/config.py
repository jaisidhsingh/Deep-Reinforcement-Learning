from bunch import Bunch

# Init Bunch object
config = Bunch({})

# Device configs
config.device = 'cuda'

# Input configs
config.image_size = 50
config.in_channels = 3

# Conv-BatchNorm-ReLU Block config
config.num_blocks = 3
config.block_configs = [
            Bunch({
                "out": 16,
                "kernel_size": 5,
                "stride": 2,
                "pad": 0
            }),
            Bunch({
                "out": 32,
                "kernel_size": 5,
                "stride": 2,
                "pad": 0
            }),
            Bunch({
                "out": 64,
                "kernel_size": 5,
                "stride": 2,
                "pad": 0
            }),
]

# Fully connected layer configs
config.num_layers = 3
config.layer_configs = [256, 128, 64]

# Hyperparamter configs
config.hyperparams = Bunch({
        "batch_size": 32,
        "epsilon_start": 0.99,
        "epsilon_end": 0.0,
        "epsilon_decay": 0.0,
        "gamma": 0.0,
        "target_update": 10
})

# Gym configs
config.env_name = "CartPole-v1"
config.max_steps = 1000
config.render_video = False
config.num_eps = 50
config.num_actions = 4

# Replay buffer configs
config.max_buffer_size = 10000
