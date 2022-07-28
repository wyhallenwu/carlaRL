import carla
import random
import queue


class CarlaENV(object):
    def __init__(self, config):
        self.client = carla.Client(config['host'], config['port'])
        self.world = self.client.get_world()
        self.world_settings = self.world.get_settings()
        settings = config['settings'] 
