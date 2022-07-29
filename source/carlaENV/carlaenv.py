import carla
import random
import queue
import yaml
from source.Agent import ActorCar

SETTING_FILE = '../config.yaml'


def get_env_settings():
    """get all the settings of the Carla Simulator.

    The settings can be configured in config.yaml

    Returns:
        a dict of the initial settings
    """
    with open(SETTING_FILE, 'r') as f:
        env_settings = yaml.safe_load(f.read())

    # settings should follow the instructions
    assert env_settings['SYN']['fixed_delta_seconds'] <= env_settings['substepping']['max_substep_delta_time'] * \
        env_settings['substepping']['max_substeps'], "substepping settings wrong!"
    return env_settings


class CarlaENV(object):
    def __init__(self):
        """initialize the environment.
        important members:
            self.config
            self.client
            self.world
            self.agent
            self.traffic_manager
        """
        self.config = get_env_settings()
        self.client = carla.Client(self.config['host'], self.config['port'])
        self.client.set_timeout(3.0)
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager(
            self.config['tm_port'])
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.config['seed'])
        self.agent = None
        self.vehicle_control = None
        # get blueprint and spawn_points
        self.bp = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        # update settings
        self._update_settings()
        self.world.apply_settings(self.world_settings)
        # refresh world
        self.client.reload_world(False)
        self._set_env()

    def _update_settings(self):
        self.world_settings = self.world.get_settings()
        if self.config['syn'] is not None and self.config['substepping'] is not None:
            self.world_settings.synchronous_mode = True
            self.world_settings.fixed_delta_seconds = self.config['syn']['fixed_delta_seconds']
            self.world_settings.substepping = True
            self.world_settings.max_substep_delta_time = self.config[
                'substepping']['max_substep_delta_time']
            self.world_settings.max_substeps = self.config['substepping']['max_substeps']

    def _set_env(self):
        """adding npc cars and create actor."""
        cars = self.bp.filter("vehicle")
        for i in range(self.config['car_num']):
            car = self.world.spawn_actor(
                random.choice(cars), self.spawn_points[i])
            car.set_autopilot(True)
        # adding agent(combination of car and camera)
        self.agent = ActorCar(self.world, self.bp, self.spawn_points)
        self.vehicle_control = self.agent.actor_car.apply_control

    def step(self, action):
        """take an action.

        Returns:
            observation(np.array(640, 480, 3))
            reward(int)
            done(bool)
        """
        assert isinstance(
            action, carla.VehicleControl), "action type is not vehicle control"
        self.vehicle_control(action)
        self.world.tick()
        observation, intensity = self.agent.retrieve_data()
        reward = self.get_reward(action, intensity)
        done = True if reward == -200 else False
        return observation, reward, done

    def reset(self):
        """reset the environment while keeping the init settings."""
        # set false to keep the settings in sync
        self._update_settings()
        self.client.reload_world(False)
        assert len(self.world.get_actors().filter(
            'vehicle')) == 0, "reload wrong"
        self._set_env()
        assert len(self.world.get_actors().filter(
            'vehicle')) != 0, "reload succeed"

    def get_reward(self, action, intensity):
        """reward policy.
        Args:
            action(carla.VehicleControl)
            intensity(float):the length of the collision_impluse
        Returns:
            reward:int
        """
        if intensity > 10:
            return -200
        elif action.hand_break or action.reverse:
            return -10
        else:
            return 1
