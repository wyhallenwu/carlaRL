import numpy as np
import yaml
# from source.carlaENV.carlaenv import CarlaEnv
import torch
import carla


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


def sample_trajectory(env, action_policy, max_episode_length):
    """Sample one trajectory."""
    ob, _ = env.reset()
    # env.set_timeout(5)
    steps = 0
    obs, acs, rws, next_obs, terminals = [], [], [], [], []
    while True:
        obs.append(ob)
        ac = action_policy.get_action(ob)
        ac = convert_tensor2control(ac)
        # ac = action_policy  # test env
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
    for _ in range(n):
        path = sample_trajectory(env, action_policy, max_episode_length)
        paths.append(path)
    return paths


def convert_path2list(paths):
    """convert the path to five list."""
    observations = [path["observations"] for path in paths]
    actions = [path["actions"] for path in paths]
    rewards = [path["rewards"] for path in paths]
    next_obs = [path["next_obs"] for path in paths]
    terminals = [path["terminals"] for path in paths]
    return observations, actions, rewards, next_obs, terminals


def convert_control2numpy(action: carla.VehicleControl) -> np.ndarray:
    """Convert the control to numpy array."""
    return np.array([action.throttle, action.steer, action.brake])


def convert_tensor2control(pred_action: torch.Tensor) -> carla.VehicleControl:
    ac = tonumpy(pred_action)
    return carla.VehicleControl(ac[0], ac[1], ac[2])


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def totensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float().to(device)


def tonumpy(x: torch.Tensor) -> np.ndarray:
    return x.to('cpu').detach().numpy()


def map2action(index):
    """map action index to action.

    Returns:
        carla.VehicleControl()
    """
    if index == 0:
        return carla.VehicleControl(1, 0, 0)
    elif index == 1:
        return carla.VehicleControl(1, -1, 0)
    elif index == 2:
        return carla.VehicleControl(1, 1, 0)
    else:
        return carla.VehicleControl(0, 0, 1)
