from source.model import ActorCritic
from source.carlaENV.carlaenv import CarlaEnv
import source.utils.util as util
from source.replaybuffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self):
        self.env = CarlaEnv()
        self.config = util.get_env_settings()
        self.ac_net = ActorCritic(
            4, self.config['hidden_dim'], self.config['n_layers'], self.config['gamma'])
        self.replaybuffer = ReplayBuffer(self.config['buffer_size'])
        self.writer = SummaryWriter()

    def train(self):
        """on-policy actor-critic training."""
        # sample n trajectories
        paths = util.sample_n_trajectories(
            self.config['sample_n'], self.env, self.ac_net, 10000)
        # add trajectories to replaybuffer
        self.replaybuffer.add_rollouts(paths)
        # sample lastest trajectories for training
        training_paths = self.replaybuffer.sample_recent_rollouts(1000)
        loss, t = self.ac_net.update(training_paths)
        return loss, t

    def check_rewards(self, i):
        check_paths = self.replaybuffer.sample_recent_rollouts(10)
        average_reward = util.check_average_reward(check_paths)
        print(f"average reward at ({i}) is: {average_reward}")

    def training_loog(self):
        for i in range(self.config['epoch']):
            if (i + 1) % 100 == 0:
                self.check_rewards(i)
            else:
                loss, t = self.train()
                self.writer.add_scalar('loss/epoch', loss, i)
                self.writer.add_scalar('time/epoch', t, i)
