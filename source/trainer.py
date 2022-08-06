from source.model import ActorCritic
from source.carlaENV.carlaenv import CarlaEnv
import source.utils.util as util
from source.replaybuffer import ReplayBuffer


class Trainer(object):
    def __init__(self):
        self.env = CarlaEnv()
        self.config = util.get_env_settings()
        self.ac_net = ActorCritic(
            4, self.config['hidden_dim'], self.config['n_layers'], self.config['gamma'])
        self.replaybuffer = ReplayBuffer(self.config['buffer_size'])

    def train(self):
        """on-policy actor-critic training."""
        # sample n trajectories
        paths = util.sample_n_trajectories(10, self.env, self.ac_net, 10000)
        # add trajectories to replaybuffer
        self.replaybuffer.add_rollouts(paths)
        # sample lastest trajectories for training
        training_paths = self.replaybuffer.sample_recent_rollouts(1000)
        self.ac_net.train(training_paths)
