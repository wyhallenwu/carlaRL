import numpy as np
import yaml
from source.carlaENV.carlaenv import CarlaEnv


def get_env_settings(filename):
    """get all the settings of the Carla Simulator.

    The settings can be configured in config.yaml

    Returns:
        a dict of the initial settings
    """
    with open(filename, 'r') as f:
        env_settings = yaml.safe_load(f.read())

    # settings should follow the instructions
    assert env_settings['syn']['fixed_delta_seconds'] <= env_settings['substepping']['max_substep_delta_time'] * \
        env_settings['substepping']['max_substeps'], "substepping settings wrong!"
    return env_settings


def Path(obs, acs, rws, next_obs, terminals):
    """wrap a episode to a Path.
    Returns:
        Path(dict):
    """
    if obs != []:
        obs = np.stack(obs, axis=0)
        acs = np.stack(acs, axis=0)
        next_obs = np.stack(next_obs, axis=0)
    return {
        "observations": np.array(obs, dtype=np.uint8),
        "actions": np.array(acs, dtype=np.float32),
        "rewards": np.array(rws, dtype=np.int),
        "next_obs": np.array(next_obs, dtype=np.uint8),
        "terminals": np.array(terminals, dtype=np.uint8)
    }


def sample_trajectory(env: CarlaEnv, action_policy, max_episode_length):
    """Sample one trajectory."""
    ob, _ = env.reset()
    steps = 0
    obs, acs, rws, next_obs, terminals = [], [], [], []
    while True:
        obs.append(ob)
        ac = action_policy.get_action(ob)
        next_ob, reward, done = env.step(ac)
        acs.append(ac)
        rws.append(reward)
        next_obs.append(next_ob)
        terminals.append(done)
        ob = next_ob
        steps += 1

        if done or steps >= max_episode_length:
            break

    return Path(obs, acs, rws, next_obs, terminals)


def sample_n_trajectories(n, env, action_policy, max_episode_length):
    paths = []
    for i in range(n):
        path = sample_trajectory(env, action_policy, max_episode_length)
        paths.append(path)

    return paths


def convert_path2list(paths):
    observations = [path["observations"] for path in paths]
    actions = [path["actions"] for path in paths]
    rewards = [path["rewards"] for path in paths]
    next_obs = [path["next_obs"] for path in paths]
    terminals = [path["terminals"] for path in paths]

    return observations, actions, rewards, next_obs, terminals
