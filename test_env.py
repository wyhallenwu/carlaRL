from source.carlaENV import carlaenv
import carla
env = carlaenv.CarlaEnv()
n = 0
while True:
    env.reset()
    observations, reward, done = env.step(carla.VehicleControl(1, 0, 0))
    n += 1
    print(n)
    for a in env.world.get_actors():
        print(a)
