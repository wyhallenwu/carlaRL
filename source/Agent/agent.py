import carla
import numpy as np


class ActorCar(object):
    """ActorCar is the combination of car and attached camera.
    important members:
        self.actor_car
        self.rgb_camera
        self.col_sensor
    """

    def __init__(self, world, bp, spawn_points):
        car = bp.filter('model3')[0]
        spawn_point = spawn_points[0]
        self.actor_car = world.spawn_actor(car, spawn_point)
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

        self.actor_list = [self.actor_car, self.rgb_camera, self.col_sensor]
        self.front_camera = None
        self.collision_intensity = None

    def retrieve_data(self):
        self.rgb_camera.listen(lambda image: self.process_img(
            image.raw_data))

        self.col_sensor.listen(
            lambda event: self.process_col_event(event))

        return self.front_camera, self.collision_intensity

    def process_img(self, raw_data):
        img = np.reshape(raw_data, (640, 480, 4))
        self.front_camera = img[:, :, 3]/255.0

    def process_col_event(self, event):
        impulse = event.normal_impulse
        self.collision_intensity = impulse.length()
