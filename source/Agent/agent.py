import carla


class ActorCar(object):
    """ActorCar is the combination of car and attached camera.
    important members:
        self.actor_car
        self.rgb_camera
    """

    def __init__(self, world, bp, spawn_points):
        self.actor_car = bp.filter('model3')[0]
        spawn_point = spawn_points[0]
        world.spawn_actor(self.actor_car, spawn_point)
        camera = bp.find('sensor.camera.rgb')
        camera.set_attribute('image_size_x', '640')
        camera.set_attribute('image_size_y', '480')
        camera.set_attribute('fov', '110')
        transform = carla.Transform(carla.Location(x=2.5, z=0.5))
        self.rgb_camera = world.spawn_actor(camera, transform,
                                            attach_to=self.actor_car)
        collision_sensor = bp.find('sensor.other.collision')
        self.col_sensor = world.spawn_actor(collision_sensor,
                                            transform, attach_to=self.actor_car)
