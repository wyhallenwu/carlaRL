from source.carlaENV import carlaenv
import carla
env = carlaenv.CarlaEnv()
while True:
    env.reset()
    observations, reward, done = env.step(carla.VehicleControl(1, 0, 0))