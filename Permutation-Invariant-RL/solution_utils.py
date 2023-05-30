import pickle
import torch
import torch.nn as nn
import numpy as np


class BaselineLinearModel(nn.Module):
    """"
    Paramters:
    ----------
    num_features: int
        number of features in the observation vector

    config: dict
        configuration of the MLP

    Attributes:
    -----------
    layers: list
        list of nn.Linear layers along with nn.Tanh activations

    net: nn.Sequential
        actual MLP model
    """

    def __init__(self, num_features, config):
        super().__init__()
        self.num_layers = config['num_layers']
        self.layer_sizes = config['layer_sizes']
        self.num_features = num_features

        layers = []

        input_size = self.num_features
        for i in range(self.num_layers):
            layers.append(nn.Linear(input_size, self.layer_sizes[i]))
            layers.append(nn.Tanh())

            input_size = self.layer_sizes[i]

        self.net = nn.Sequential(*layers)

        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        """"
        Parameters:
        -----------
        x: torch.Tensor
            our observation vector

        Returns:
        --------
        self.net(x): torch.Tensor
        """
        return self.net(x)


class QueryPositionalEncodings():
    """
    Parameters:
    -----------
    n: int
        number of rows of the query table

    dim: int
        number of columns of the query table

    Attributes:
    -----------
    encoding: np.ndarray
        initialized as None, stores the table for positional encodings
    """

    def __init__(self, n, dim):
        self.encoding = None
        self.n = n
        self.dim = dim

    def get_angle(self, x, k):
        return x / np.power(10000, 2*(k//2)/self.dim)

    def get_angle_row(self, x):
        return [self.get_angle(x, i) for i in range(self.dim)]

    def get_encodings(self):
        encoding_table = np.array([self.get_angle_row(i)
                                  for i in range(self.n)]).astype(float)
        encoding_table[:, 0::2] = np.sin(encoding_table[:, 0::2])
        encoding_table[:, 1::2] = np.cos(encoding_table[:, 1::2])

        return torch.from_numpy(encoding_table)


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, project_dims, scaling=True):
        super().__init__()
        self.project_queries = nn.Linear(hidden_size, project_dims, bias=False)
        self.project_keys = nn.Linear(hidden_size, project_dims, bias=False)

        if scaling:
            self.scaling = torch.sqrt(project_dims)
        else:
            self.scaling = 1

    def forward(self, queries, keys):
        # queries shape: (N, H), keys shape: (M, H)
        queries, keys = self.project_queries(queries), self.project_keys(keys)
        # queries shape: (N, D), keys shape: (M, D)

        attention_product = queries @ keys.T
        # attention product shape: (N, M)

        scaled_attention = torch.div(attention_product, self.scaling)
        attention_weights = torch.tanh(scaled_attention)

        return attention_weights


class PermutationInvariantLayer(nn.Module):
    def __init__(self, embedding_dim, project_dim, hidden_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.project_dim = project_dim
        self.hidden_size = hidden_size

        self.hidden_state_tuple = None
        self.lstm = nn.LSTMCell(input_size=2, hidden_size=hidden_size)

        self.attention = AttentionBlock(
            hidden_size=self.hidden_size,
            project_dims=self.project_dim,
            scaling=False
        )

        self.Q = QueryPositionalEncodings(
            embedding_dim, hidden_size).get_encodings().float()

    def forward(self, x, previous_action):
        # x is the current observation of shape (5,)

        num_features = len(x)
        previous_action = float(previous_action)

        predicted_action = torch.cat(
            [
                x[:, None],
                torch.ones(num_features, 1)*previous_action,
            ],
            dim=-1
        )  # make shape (num_features, 2)

        if self.hidden_state_tuple is None:
            self.hidden_state_tuple = (
                torch.zeros(num_features, self.hidden_size),
                torch.zeros(num_features, self.hidden_size),
            )
        self.hidden_state_tuple = self.lstm(
            predicted_action, self.hidden_state_tuple
        )

        queries = self.Q
        keys = self.hidden_state_tuple[0]
        values = x.unsqueeze(1)

        attention_weights = self.attention(queries, keys)
        # shape of attention weights: (embedding_dim, num_features)

        latent_value = torch.tanh(attention_weights @ values).squeeze()
        # latent value shape: (emebdding_dim, 1) squeezed to (embedding_dim,)

        return latent_value, attention_weights


class PermutationInvariantPolicyNetwork(nn.Module):
    def __init__(self, embedding_dim, project_dim, hidden_size):
        super().__init__()
        self.sensory_neuron = PermutationInvariantLayer(
            embedding_dim,
            project_dim,
            hidden_size=hidden_size,
        )

        self.projection = nn.Linear(embedding_dim, 1)

        for parameter in self.parameters():
            parameter.requries_grad = False

    def forward(self, x, previous_action):
        latent_value, attention_weights = self.sensory_neuron(
            x, previous_action)
        prediction = torch.tanh((self.projection(latent_value.unsqueeze(0))))

        return prediction[0]


