from tokenize import String
import torch
import torch.nn as nn
import torch.optim as optim
import source.utility as util

config = util.get_env_settings("./config.yaml")
device = util.device


class ValueNetwork(nn.Module):
    def __init__(self) -> None:
        super(ValueNetwork, self).__init__()
        self.optimizer = optim.Adam(
            self.parameters(), lr=config['valuenet_lr'])
        self.resnet = util.build_resnet()
        self.layers = self.build_layers()
        self.loss_fn = nn.MSELoss()

    def build_layers(self):
        layers = []
        layers.append(
            nn.Linear(self.resnet.fc.out_features, config['hidden_dim']))
        for _ in range(config['n_layers']):
            layers.append(
                nn.Linear(config['hidden_dim'], config['hidden_dim']))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config['hidden_dim'], 1))
        return nn.Sequential(*layers)

    def forward(self, obs):
        obs = self.resnet(obs.to(device))
        value = self.layers(obs)
        return value

    def update(self):
        pass


class SoftQNet(nn.Module):
    def __init__(self, action_dim) -> None:
        super(SoftQNet, self).__init__()
        self.optimizer = optim.Adam(self.parameters(), lr=config['softq_lr'])
        self.resnet = util.build_resnet()
        self.layers = self.build_layers()
        self.action_dim = action_dim

    def build_layers(self):
        layers = []
        layers.append(nn.Linear(self.resnet.fc.out_features +
                      self.action_dim, config['hidden_dim']))
        for _ in range(config['n_layers']):
            layers.append(
                nn.Linear(config['hidden_dim'], config['hidden_dim']))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config['hidden_dim'], 1))
        return nn.Sequential(*layers)

    def forward(self, obs):
        obs = self.resnet(obs.to(device))
        result = self.layers(obs)
        return result


class PolicyNet(nn.Module):
    def __init__(self, action_dim, log_min, log_max):
        super(PolicyNet, self).__init__()
        self.optimizer = optim.Adam(self.parameters(), lr=config['policy_lr'])
        self.resnet = util.build_resnet()
        self.std_layer = self.build_layers()
        self.mean_layer = self.build_layers()
        self.action_dim = action_dim
        self.log_min = log_min
        self.log_max = log_max

    def build_layers(self):
        layers = []
        layers.append(
            nn.Linear(self.resnet.fc.out_features, config['hidden_dim']))
        for _ in range(config['n_layers']):
            layers.append(
                nn.Linear(config['hidden_dim'], config['hidden_dim']))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config['hidden_dim'], self.action_dim))
        return nn.Sequential(*layers)

    def forward(self, obs):
        obs = self.resnet(obs.to(device))
        mean = self.mean_layer(obs)
        log_std = self.std_layer(obs)
        log_std = torch.clamp(log_std, self.log_min, self.log_max)
        return mean, log_std

    