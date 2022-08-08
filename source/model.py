import torch
import torch.nn as nn
from torchvision import models, transforms
from torch import distributions
from torch import optim
import numpy as np
import time
import source.utility as util
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class ActorCritic(nn.Module):
    def __init__(self, ac_dim, hidden_dim, n_layers, gamma, learning_rate):
        super(ActorCritic, self).__init__()
        self.ac_dim = ac_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.resnet = models.resnet50(pretrained=True)
        # using pretrained resnet50
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.layers = self.build_layers()
        self.actor_layer = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.ac_dim),
            nn.Softmax(dim=1),
        ])
        self.critic_layer = nn.Linear(self.hidden_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def process_imgs(self, imgs):
        """process_imgs processes PIL images with Resnet50 and return a mini-batch tensor."""
        if not isinstance(imgs, list):
            imgs = self.image_transform(imgs)
            return imgs.unsqueeze(0)
        else:
            return torch.stack([self.image_transform(img).unsqueeze(0) for img in imgs])

    def build_layers(self):
        layers = []
        layers.append(nn.Linear(self.resnet.fc.out_features, self.hidden_dim))
        for _ in range(self.n_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, obs):
        obs = obs.to(device)
        input = self.resnet(obs)
        middle_result = self.layers(input)
        probs = self.actor_layer(middle_result).squeeze()
        v_value = self.critic_layer(middle_result).squeeze()
        actions_distribution = distributions.Categorical(probs)
        return actions_distribution, v_value

    def get_action(self, obs):
        print(f"obs shape: {obs.shape}")
        obs = obs.unsqueeze(0).to(device)
        action_prob, _ = self.forward(obs)
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

    def update(self, paths, epoch_i):
        observations, actions, rewards, next_obs, terminals, frames = util.convert_path2list(
            paths)
        start = time.time()
        loss_list = []
        for i in range(len(paths)):
            obs, acs, rws, nextobs, terminal = observations[
                i], actions[i], rewards[i], next_obs[i], terminals[i]
            obs = obs.to(device)
            nextobs = nextobs.to(device)
            # update critic
            print("fit v model.")
            _, v_current = self.forward(obs)
            self.optimizer.zero_grad()
            _, v_next = self.forward(nextobs)
            target = self.gamma * v_next + util.totensor(rws)
            critic_loss = self.loss_fn(v_current, target)
            critic_loss.backward()
            self.optimizer.step()
            print("fit v model done.")
            # update actor
            print("update actor")
            self.optimizer.zero_grad()
            pred_action, v_value = self.forward(obs)
            advantages = self.compute_advantage(
                obs, rws, terminal, util.tonumpy(v_value))
            loss = -torch.mean(pred_action.log_prob(acs)
                               * util.totensor(advantages))
            loss.backward()
            self.optimizer.step()
            print(f"loss: {loss}")
            loss_list.append(util.tonumpy(loss))
            print("update actor done.")
        end = time.time()
        util.log_training(np.mean(loss_list), epoch_i)
        print(f"time: {end - start}")
