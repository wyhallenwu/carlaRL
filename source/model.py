import torch
import torch.nn as nn
from torchvision import models, transforms
from torch import distributions
from torch import optim
import time
import random
import source.utils.util as util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, ac_dim, hidden_dim, n_layers, gamma):
        super(ActorCritic, self).__init__()
        self.ac_dim = ac_dim
        # self.ob_dim = ob_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gamma = gamma
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        self.resnet = models.resnet50(pretrained=True)
        # using pretrained resnet50
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.layers = self.build_layers()
        self.actor_layer = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.ac_dim),
            nn.Softmax(),
        ])
        self.critic_layer = nn.Linear(self.hidden_dim, 1)

    def process_imgs(self, imgs):
        """process_imgs processes PIL images with Resnet50 and return a mini-batch tensor."""
        images = []
        for img in imgs:
            images.append(self.image_transform(img))
        return torch.stack(images)

    def build_layers(self):
        layers = []
        layers.append(nn.Linear(self.resnet.fc.out_features, self.hidden_dim))
        for _ in range(self.n_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())
        # layers.append(nn.Linear(self.hidden_dim, self.ac_dim))
        # layers.append(nn.Softmax())
        return nn.Sequential(*layers)

    def forward(self, images):
        imgs = self.process_imgs(images)
        input = self.resnet(imgs)
        middle_result = self.layers(input)
        probs = self.actor_layer(middle_result).squeeze()
        v_value = self.critic_layer(middle_result)
        actions_distribution = distributions.Categorical(probs)
        return actions_distribution, v_value.squeeze()

    def get_action(self, obs):
        action_prob, _ = self.forward(obs)
        # shape [1, batch]
        action = action_prob.sample()
        index = util.tonumpy(action)[0]
        assert isinstance(index, int), "type of action index is not int."
        return index

    def compute_advantage(self, obs, rws, terminals):
        # compute q_value(TD)
        pass
    def train(self, paths):
        observations, actions, rewards, next_obs, terminals = util.convert_path2list(
            paths)
        pred_action, v_value = self.forward(observations)
