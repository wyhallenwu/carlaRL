# README
this the final project of RL course. The project is to train a simple autonomous vehicle in Carla Simulator.    

## todos
- [x] wrap the environment of carla following the paradigm of OpenAI gym
  - [x] env() init the world
  - [x] step() return info
  - [x] reset() reset the world to the init status
  - [x] agent(actor)

> need to fix problem of reset environment. May using destroy() for all actors

> solution:
> use collision to indicate the episode ends.

> receive warning when destroy sensors: you should firstly sensor.stop()
> don't use reload_world(), it causes some problems(high memory usage and finally core dumped)

- [ ] sample trajectories
- [ ] replaybuffer