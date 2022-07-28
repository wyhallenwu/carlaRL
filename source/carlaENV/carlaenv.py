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
            self.client
            self.world
            self.agent
            self.npc_cars
            self.traffic_manager
        """
        config = get_env_settings()
        self.client = carla.Client(config['host'], config['port'])
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True) 
        self.traffic_manager.set_random_device_seed(0)
        # settings update
        self.world_settings = self.world.get_settings()
        if config['syn'] is not None and config['substepping'] is not None:
            self.world_settings.synchronous_mode = True
            self.world_settings.fixed_delta_seconds = config['syn']['fixed_delta_seconds']
            self.world_settings.substepping = True
            self.world_settings.max_substep_delta_time = config['substepping']['max_substep_delta_time']
            self.world_settings.max_substeps = config['substepping']['max_substeps']
        # init all settings
        self.world.apply_settings(self.world_settings)
        # adding objects and set traffic manager
        bp = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        self.npc_cars = bp.filter('vehicle')[1:51]
        for car in self.npc_cars:
            car.set_autopilot(True)
        npc_spawn_points = spawn_points[1:51]
        self.world.spawn_actor(self.npc_cars, npc_spawn_points)
        # adding agent(combination of car and camera)
        self.agent = ActorCar(self.world, bp, spawn_points)
        
        
    def step(self, action):
        pass


    