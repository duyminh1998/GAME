from gym.envs.registration import register

register(
    id='MountainCar3D-v0',
    entry_point='GAME.envs.mountain_car.3d_mountain_car:MountainCar3DEnv',
    max_episode_steps=300,
)