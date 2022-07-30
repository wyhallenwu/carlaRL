import carla
import random
import queue
import yaml
from source.Agent.agent import ActorCar

SETTING_FILE = "./config.yaml"


def get_env_settings():
    """get all the settings of the Carla Simulator.

    The settings can be configured in config.yaml

    Returns:
        a dict of the initial settings
    """
    with open(SETTING_FILE, 'r') as f:
        env_settings = yaml.safe_load(f.read())

    # settings should follow the instructions
    assert env_settings['syn']['fixed_delta_seconds'] <= env_settings['substepping']['max_substep_delta_time'] * \
        env_settings['substepping']['max_substeps'], "substepping settings wrong!"
    return env_settings


class CarlaEnv(object):
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
        self.client.set_timeout(15)
        self.world = self.client.get_world()
        # self.traffic_manager = self.client.get_trafficmanager(
        #     self.config['tm_port'])
        # self.traffic_manager.set_synchronous_mode(True)
        # self.traffic_manager.set_random_device_seed(self.config['seed'])
        self.agent = None
        self.vehicle_control = None
        self.actor_list_env = []
        # get blueprint and spawn_points
        self.bp = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        # update settings
        self._update_settings()
        self.world.apply_settings(self.world_settings)
        print("init actors num", len(self.world.get_actors().filter(
            'vehicle')))
        # refresh world
        # self.client.reload_world(False)
        # self._set_env()

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
        print(f"set {self.config['car_num']} vehicles in the world")
        for i in range(self.config['car_num']):
            car = self.world.spawn_actor(
                random.choice(cars), self.spawn_points[i + 1])
            # self.client.set_timeout(1)
            car.set_autopilot(True)
            self.actor_list_env.append(car)
        print(f"setting {len(self.actor_list_env)} in _set_env")
        # adding agent(combination of car and camera)
        self.agent = ActorCar(self.client, self.world,
                              self.bp, self.spawn_points)
        self.vehicle_control = self.agent.actor_car.apply_control

    def step(self, action):
        """take an action.
        Args:
            action(carla.VehicleControl):throttle, steer, break, hand_break, reverse
        Returns:
            observation(np.array(640, 480, 3))
            reward(int)
            done(bool)
        """
        assert isinstance(
            action, carla.VehicleControl), "action type is not vehicle control"
        self.vehicle_control(action)
        frame_index = self.world.tick()
        print(f"after step, current frame is: {frame_index}")
        observation, collision = self.agent.retrieve_data(frame_index)
        reward = self.get_reward(action, collision)
        done = True if collision != 0 else False
        return observation, reward, done

    def reset(self):
        """reset the environment while keeping the init settings."""
        # set false to keep the settings in sync
        print("initialize environment.")
        self.cleanup_world()
        # self.client.reload_world(False)
        self.client.set_timeout(15)
        # adding cars to env
        # self.world.tick()
        self._update_settings()
        self._set_env()
        # deploy env in sync mode
        frame_index = self.world.tick()
        print(f"after reset, current frame is: {frame_index}")
        # self.client.set_timeout(10)
        # print("check: ", len(self.world.get_actors().filter(
        #     '*vehicle*')))
        assert len(self.world.get_actors().filter(
            '*vehicle*')) == (self.config['car_num'] + 1), "set env wrong"
        # return start image frame
        return self.agent.retrieve_data(frame_index)

    def get_reward(self, action, intensity):
        """reward policy.
        Args:
            action(carla.VehicleControl)
            intensity(float):the length of the collision_impluse
        Returns:
            reward:int
        """
        if intensity != 0:
            return -200
        elif action.hand_brake or action.reverse:
            return -10
        else:
            return 1

    def cleanup_world(self):
        # clean up the env
        print("actorlist length: ", len(self.actor_list_env))
        # for actor in self.actor_list_env:
        #     assert actor.destroy(), "destroy actor false in env"
        self.client.apply_batch([carla.command.DestroyActor(x)
                                 for x in self.actor_list_env])
        # clean up the agent
        if self.agent is not None:
            print("destroy agent")
            self.agent.cleanup()
        self.agent = None
        self.actor_list_env = []
        print("clean up the world")
        print("after cleanup world actors: ", len(self.world.get_actors().filter(
            'vehicle')))
        assert len(self.world.get_actors().filter(
            'vehicle')) == 0, "cleanup world wrong"

    def get_all_actors(self):
        """get all Actors in carla env.
        Returns:
            carla.ActorList
        """
        return self.world.get_actors()

    def get_all_vehicles(self):
        """get all vehicles in carla env including actor_car.
        Returns:
            carla.ActorList
        """
        return self.world.get_actors().filter('vehicle')

    def _exit(self):
        # self.client.reload_world(False)
        self.cleanup_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        print(
            f"before exited, there are { len(self.get_all_vehicles())} actors")
        print("exit world")
