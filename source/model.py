import torch
import torch.nn as nn
from torchvision import models, transforms
from torch import distributions
from torch import optim
import numpy as np
import time
import source.utility as util

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
        self.optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])
        self.loss_fn = nn.MSELoss()

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
        return action

    def compute_advantage(self, obs, rws, terminals, v_current: np.ndarray):
        advantages = np.zeros(obs.shape[0])
        # compute q_value(TD)
        for i in range(len(v_current)):
            if terminals[i] == 1:
                advantages[i] = rws[i] - v_current[i]
            else:
                advantages[i] = rws[i] + self.gamma * \
                    v_current[i + 1] - v_current[i]
        return advantages

    def update(self, paths):
        start = time.time()
        observations, actions, rewards, next_obs, terminals = util.convert_path2list(
            paths)
        # update critic
        _, v_current = self.forward(observations)
        self.optimizer.zero_grad()
        _, v_next = self.forward(next_obs)
        target = self.gamma * v_next + util.totensor(rewards)
        critic_loss = self.loss_fn(v_current, target)
        critic_loss.backward()
        self.optimizer.step()
        # update actor
        self.optimizer.zero_grad()
        pred_action, v_value = self.forward(observations)
        advantages = self.compute_advantage(
            observations, rewards, terminals, util.tonumpy(v_value))
        loss = -torch.mean(pred_action.log_prob(actions)
                           * util.totensor(advantages))
        loss.backward()
        self.optimizer.step()
        end = time.time()
        return util.tonumpy(loss), end-start
