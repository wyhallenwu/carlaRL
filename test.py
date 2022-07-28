import carla

world = carla.get_world()
vehicles = world.get_blueprint_library().filter('vehicle')
print(vehicles)
