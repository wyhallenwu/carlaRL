import carla
import numpy as np
import queue


class ActorCar(object):
    """ActorCar is the combination of car and attached camera.
    important members:
        self.actor_car
        self.rgb_camera
        self.col_sensor
    """

    def __init__(self, client, world, bp, spawn_points):
        self.client = client
        self.actor_list = []
        car = bp.filter('model3')[0]
        spawn_point = spawn_points[0]
        self.actor_car = world.spawn_actor(car, spawn_point)
        self.actor_list.append(self.actor_car)
        camera = bp.find('sensor.camera.rgb')
        camera.set_attribute('image_size_x', '640')
        camera.set_attribute('image_size_y', '480')
        camera.set_attribute('fov', '110')
        transform = carla.Transform(carla.Location(x=1.2, z=1.7))
        self.rgb_camera = world.spawn_actor(camera, transform,
                                            attach_to=self.actor_car)
        self.actor_list.append(self.rgb_camera)
        # tips: collision sensor only receive data when triggered
        collision_sensor = bp.find('sensor.other.collision')
        self.col_sensor = world.spawn_actor(collision_sensor,
                                            transform, attach_to=self.actor_car)
        self.actor_list.append(self.col_sensor)

        self.front_camera = None
        self.collision_intensity = 0
        self._camera_queue = queue.Queue()
        self._col_queue = queue.Queue()
        self.rgb_camera.listen(self._camera_queue.put)
        self.col_sensor.listen(self._col_queue.put)

    def retrieve_data(self, frame_index):
        while not self.process_img(frame_index):
            pass
        # self.process_img(frame_index)
        self.process_col_event(frame_index)
        return self.front_camera, self.collision_intensity

    def process_img(self, frame_index):
        if not self._camera_queue.empty():
            image = self._camera_queue.get(timeout=2)
            # print("current size of q images", self._col_queue.qsize())
            assert self._camera_queue.qsize() == 0, "Expected qsize of images 0"
            assert frame_index == image.frame, "not the corresponding frame image."
            print(f"current image frame is: {image.frame}")
            img = np.reshape(image.raw_data, (640, 480, 4))
            img = img[:, :, :3]
            # normalize the image to [0,1]
            self.front_camera = img[:, :, ::-1]/255.0
            return True
        self.front_camera = None
        return False

    def process_col_event(self, frame_index):
        if not self._col_queue.empty():
            event = self._col_queue.get(timeout=2)
            assert frame_index == event.frame, "not the corresponding frame event."
            impulse = event.normal_impulse
            self.collision_intensity = impulse.length()
            print(f"collision length is: {self.collision_intensity}")
            print(
                f"current collision frame is: {event.frame}")

    def cleanup(self):
        """cleanup is to destroy all agent actors in the world."""
        # for a in self.actor_list:
        #     assert a.destroy(), "destroy actor wrong in agent "
        self.rgb_camera.stop()
        self.col_sensor.stop()
        self.client.apply_batch([carla.command.DestroyActor(x)
                                 for x in self.actor_list])
        print("destroy all actors of agent")
