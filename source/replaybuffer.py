from source.utils.util import convert_path2list


class ReplayBuffer(object):
    def __init__(self) -> None:
        self.paths = []
        self.observations = None
        self.rewards = None
        self.next_observations = None
        self.terminals = None

    def add_rollouts(self, paths):
        """add serveral paths to rollouts"""
        for path in paths:
            self.paths.append(path)

        observations, actions, rewards, next_obs, terminals = convert_path2list(
            paths)
