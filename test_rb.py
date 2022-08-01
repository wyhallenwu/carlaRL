from source.utils.util import sample_n_trajectories
from source.replaybuffer import ReplayBuffer
from source.carlaENV.carlaenv import CarlaEnv
import carla
env = CarlaEnv()
# try:
#     env.client.reload_world(False)
#     rb = ReplayBuffer(10000)
#     paths = sample_n_trajectories(3, env, carla.VehicleControl(1, 0, 0), 1000)
#     rb.add_rollouts(paths)
#     print(rb.get_paths_num())
#     env._exit()
# except:
#     print("error occurs")
#     env._exit()

env.client.reload_world(False)
env.client.set_timeout(15)
rb = ReplayBuffer(10000)
paths = sample_n_trajectories(3, env, carla.VehicleControl(1, 0, 0), 1000)
rb.add_rollouts(paths)
print(rb.get_paths_num())
env._exit()
