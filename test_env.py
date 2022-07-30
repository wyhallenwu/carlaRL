from source.carlaENV import carlaenv
import carla
env = carlaenv.CarlaEnv()
# print(len(env.bp.filter("vehicle")))
# print(len(env.bp.filter("*vehicle*")))
n = 0
x = 0
# print(env.world.get_map().name)


# print("start_image frame: ", )
while x < 3:
    env.client.reload_world(False)
    env.client.set_timeout(15)
    print(len(env.world.get_actors()))
    print(env)
    start_image, _ = env.reset()
    print(f"episode {x + 1}")
    while n < 1000:
        observations, reward, done = env.step(carla.VehicleControl(1, 0, 0))
        n += 1
        print(f"n: {n}")
        print(f"actor num: {len(env.world.get_actors())}")
        print(f"reward is {reward}")
        if done:
            print("done")
            break
    env.client.set_timeout(5)

    x += 1

env._exit()
