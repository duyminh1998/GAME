from gym.envs.registration import register

register(
    id='MountainCar3D-v0',
    entry_point='GAME.envs.mountain_car.3d_mountain_car:MountainCar3DEnv',
    max_episode_steps=300,
)

register(
    id='MountainCar2D-v0',
    entry_point='GAME.envs.mountain_car.2d_mountain_car:MountainCar2DEnv',
    max_episode_steps=300,
)