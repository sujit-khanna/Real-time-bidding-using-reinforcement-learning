import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


#CNN dimension calculations: https://towardsdatascience.com/the-most-intuitive-and-easiest-guide-for-convolutional-neural-network-3607be47480


"""
Convolution layer dimensions = [(image_dim + 2*padding - filter_size)/stride + 1]*[(image_dim + 2*padding - filter_size)/stride + 1]*num_filters
pooling_layer_dim = Convolution layer dimensions/pooling_size for shape 0 and 1; num_filters uneffected
"""

HID_SIZE = 128
HID_SIZE_2 = 64
HID_SIZE_3 = 32
HID_SIZE_4 = 16
LSTM_HIDDEN_SIZE = 10
class ModelA2C(nn.Module):
    """
    The network has three heads, instead of the normal two for a discrete variant of A2C.
    The first two heads return the mean value and the variance of the actions, while the
    last is the critic head returning the value of the state.
    """
    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            # nn.Sigmoid(),
            nn.Tanh(),
            nn.Linear(HID_SIZE, HID_SIZE_2),
            nn.Tanh(),
            nn.Linear(HID_SIZE_2, HID_SIZE_3),
            nn.Tanh(),
        )
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE_3, act_size),
            # nn.Tanh(), #change tbis to sigmoid or any fucntion that returs value between 0 and 1
            nn.Sigmoid()
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE_3, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(HID_SIZE_3, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), \
               self.value(base_out)


class LSTMModelA2C(nn.Module):
    """
    The network has three heads, instead of the normal two for a discrete variant of A2C.
    The first two heads return the mean value and the variance of the actions, while the
    last is the critic head returning the value of the state.
    """
    def __init__(self, obs_size, act_size, lstm_units=20, num_layers=2):
        super(LSTMModelA2C, self).__init__()
        self.lstm_units = lstm_units
        self.base = nn.LSTM(input_size=obs_size, hidden_size=self.lstm_units, num_layers=num_layers)

        self.mu = nn.Sequential(
            nn.Linear(self.lstm_units, LSTM_HIDDEN_SIZE),
            # nn.Tanh(), #change tbis to sigmoid or any fucntion that returs value between 0 and 1
            nn.Sigmoid(),
            nn.Linear(LSTM_HIDDEN_SIZE, act_size),
            nn.Sigmoid()
        )
        self.var = nn.Sequential(
            nn.Linear(self.lstm_units, act_size),
            nn.Softplus(),
        )
        self.value = nn.Sequential(
            nn.Linear(self.lstm_units, LSTM_HIDDEN_SIZE),
            # nn.Tanh(), #change tbis to sigmoid or any fucntion that returs value between 0 and 1
            nn.Sigmoid(),
            nn.Linear(LSTM_HIDDEN_SIZE, act_size)
        )

    def forward(self, x):
        base_out, _ = self.base(x)
        return self.mu(base_out), self.var(base_out), \
               self.value(base_out)


class CNNModelA2C(nn.Module):
    """
    The network has three heads, instead of the normal two for a discrete variant of A2C.
    The first two heads return the mean value and the variance of the actions, while the
    last is the critic head returning the value of the state.
    """
    def __init__(self, obs_size, act_size):
        super(CNNModelA2C, self).__init__()
        conv_dim = obs_size.shape[0]//(2**2) + 1

        self.base = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2),
            nn.ReLU(),
            nn.AvgPool1d(2,2),
            nn.Conv1d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.AvgPool1d(2, 2),
            nn.Flatten(),
            nn.Linear(1*1*32, 64),
            nn.Sigmoid(),
            nn.Linear(64, 16),
            nn.Sigmoid(),
        )

        self.mu = nn.Sequential(
            nn.Linear(96, act_size),
            # nn.Tanh(), #change tbis to sigmoid or any fucntion that returs value between 0 and 1
            nn.Sigmoid(),

        )

        self.var = nn.Sequential(
            nn.Linear(96, act_size),
            nn.Softplus(),
        )

        self.value = nn.Linear(96, act_size)


    def forward(self, x):
        prev_shape = x.shape
        if prev_shape[0]>1:
            x = x.view(prev_shape[0], prev_shape[1], 1, prev_shape[2])
        else:
            x = x.view(prev_shape[1], prev_shape[0], prev_shape[2])

        base_out = self.base(x)
        base_out = base_out.view(-1, 6*16)
        return self.mu(base_out), self.var(base_out), \
               self.value(base_out)


class CNNLSTMModelA2C(nn.Module):
    """
    The network has three heads, instead of the normal two for a discrete variant of A2C.
    The first two heads return the mean value and the variance of the actions, while the
    last is the critic head returning the value of the state.
    """
    def __init__(self, obs_size, act_size):
        super(CNNLSTMModelA2C, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            # nn.Sigmoid(),
            nn.Tanh(),
            nn.Linear(HID_SIZE, HID_SIZE_2),
            nn.Tanh(),
            nn.Linear(HID_SIZE_2, HID_SIZE_3),
            nn.Tanh(),
        )
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE_3, act_size),
            # nn.Tanh(), #change tbis to sigmoid or any fucntion that returs value between 0 and 1
            nn.Sigmoid()
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE_3, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(HID_SIZE_3, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), \
               self.value(base_out)


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        # if len(states_v.shape)>2:
        #     states_v = states_v.resize(1, 7)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, 0, 1) #change range to 0 and 1
        return actions, agent_states