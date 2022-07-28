import carla


class ActorCar(object):
    """ActorCar is the combination of car and attached camera.
    important members:
        self.actor_car
        self.rgb_camera
    """

    def __init__(self, world, bp, spawn_points):
        self.actor_car = bp.filter('model3')
        spawn_point = spawn_points[0]
        world.spawn_actor(self.actor_car, spawn_point)
        self.rgb_camera = bp.find('sensor.camera.rgb')
        camera_init_transform = carla.Transform(carla.Location(z=2))
        world.spawn_actor(self.rgb_camera, camera_init_transform,
                          attach_to=self.actor_car)

    def apply_control(self, action):
        self.actor_car.apply_control(action)
