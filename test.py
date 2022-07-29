import carla
import random

client = carla.Client('localhost', 2000)
client.set_timeout(15)
world = client.get_world()
client.reload_world(False)
car = world.get_blueprint_library().filter('*vehicle*')
# print(car)
settings = world.get_settings()
print(len(world.get_actors()))
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
settings.substepping = True
settings.max_substep_delta_time = 0.01
settings.max_substeps = 10
world.apply_settings(settings)
while True:
    settings.synchronous_mode = False
    world.apply_settings(settings)
    client.reload_world(False)
    client.set_timeout(15)
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)
    # client.reload_world(False)
    print(len(world.get_actors()))
    spawn_points = world.get_map().get_spawn_points()
    for i in range(50):
        c = world.spawn_actor(random.choice(car), spawn_points[i])
        c.set_autopilot(True)

    # car.set_autopilot(True)
    # world.tick()
    n = 0
    while n < 10:
        world.tick()
        v = world.get_actors().filter('*vehicle*')[0]
        print(v.get_control())
        print(n)
        n += 1
    # for v in world.get_actors().filter('*vehicle*'):
    #     print("v is: ", v)

    actors = world.get_actors()
    print(len(actors))
