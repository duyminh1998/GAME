# Author: Minh Hua
# Date: 11/1/2022
# Purpose: This module sets up an entire Mountain Car experiments for a certain number of episodes and trials.

import GAME.envs.mountain_car
import gym
from GAME.agents.sarsa_lambda import SarsaLambdaCMAC3DMountainCar
from GAME.utils.helper_funcs import *
from GAME.utils.data_miners import *

def MountainCar3DExperiment(
    base_agent_class,
    base_agent,
    decay_agent_eps:float=None,
    max_episodes_per_trial:int=50,
    num_trials:int=1,
    update_agent:bool=True,
    start_learning_after:int=-1,
    print_debug:bool=False,
    save_sample_data:bool=False,
    save_sample_every:int=10,
    sample_data_col_names:list=None,
    sample_data_folder:str=None,
    sample_data_filename:str=None,
    data_collector=None,
    save_agent:bool=True,
    save_agent_every:int=10,
    save_agent_folder:str=None,
    save_agent_filename:str=None,
    eval_agent:bool=False,
    save_eval_every:int=10,
    eval_data_col_names:list=None,
    save_eval_folder:str=None,
    save_eval_filename:str=None,
    eval_data_collector=None,
    env_name:str='MountainCar3D-v0',
    env_max_steps:int=5000,
    rd_seed:int=42
):
    # make the environment
    env = gym.make(env_name)
    env._max_episode_steps = env_max_steps
    observation, info = env.reset(seed=rd_seed)

    # evaluation metric
    average_steps_per_trial = [] # eval metric

    for trial in range(num_trials):
        # reset the agent each trial
        agent = base_agent_class(base_agent.alpha, base_agent.lamb, base_agent.gamma, base_agent.method, base_agent.epsilon, base_agent.num_of_tilings, base_agent.max_size)
        # save total steps for each trial
        total_steps = 0
        for ep in range(max_episodes_per_trial):
            steps = 0
            while True:
                try:
                    # current state
                    current_state = observation # [x, x_dot, y, y_dot]
                    action = agent.choose_action_eps_greedy(current_state)
                    # next state
                    observation, reward, terminated, truncated, info = env.step(action)
                    # next action
                    next_action = agent.choose_action_eps_greedy(observation)
                    # env.render()

                    # update agent
                    if update_agent and ep >= start_learning_after:
                        target = reward + agent.get_value(observation, next_action)
                        active_tiles = agent.get_active_tiles(current_state, action)
                        agent.update(active_tiles, target)

                    # save data
                    if save_sample_data:
                        transition_data = [trial, ep, steps, current_state[0], current_state[1], current_state[2], current_state[3], action, reward, observation[0], observation[1], observation[2], observation[3], next_action]
                        data_collector.log_data(transition_data)

                    # prep the next iteration
                    steps += 1

                    # reset the training
                    if terminated or truncated:
                        observation, info = env.reset()
                        total_steps += steps
                        if print_debug:
                            print("Trial: {}, Episode: {}, Number of steps: {}, Total steps: {}".format(trial, ep, steps, total_steps))
                        break
                except KeyboardInterrupt:
                    env.close()
                    if save_sample_data:
                        data_collector.export_data(sample_data_folder, sample_data_filename)
                    if eval_agent:
                        eval_data_collector.export_data(save_eval_folder, save_eval_filename)
                    # save agent
                    if save_agent:
                        save_agent_to_file(agent, save_agent_folder, save_agent_filename)

            # save data every few iterations
            if save_sample_every and (ep % save_sample_every == 0):
                 data_collector.export_data(sample_data_folder, sample_data_filename)
            if save_agent and (ep % save_agent_every == 0):
                # save agent
                save_agent_to_file(agent, save_agent_folder, save_agent_filename)

            # at the end of each episode, we must evaluate the agent without exploration
            prev_eps = agent.epsilon
            if eval_agent:
                agent.epsilon = 0
                total_rewards = 0
                while True:
                    current_state = observation # [x, x_dot]
                    action = agent.choose_action_eps_greedy(current_state)
                    # next state
                    observation, reward, terminated, truncated, info = env.step(action)
                    # count the number of steps incurred
                    total_rewards += reward
                    # reset the training
                    if terminated or truncated:
                        observation, info = env.reset()
                        break
                # after we finish evaluation, save the evaluation data
                data_dict = {
                    col_name : col_data for col_name, col_data in zip(eval_data_col_names, [trial, ep, total_rewards])
                }
                eval_data_collector.log_data(data_dict)
                # save data every few iterations
                if (ep % save_eval_every == 0):
                    eval_data_collector.export_data(save_eval_folder, save_eval_filename)
            # at the end of each episode, decay the agent's epsilon rate
            if decay_agent_eps:
                agent.epsilon = prev_eps * decay_agent_eps
        # update trial's total step
        average_steps_per_trial.append(total_steps / max_episodes_per_trial)

    # training complete
    env.close()
    if save_sample_data:
        data_collector.export_data(sample_data_folder, sample_data_filename)
    if eval_agent:
        eval_data_collector.export_data(save_eval_folder, save_eval_filename)
    if save_agent:
        save_agent_to_file(agent, save_agent_folder, save_agent_filename)
    
    # return the average steps per trial
    return average_steps_per_trial

if __name__ == "__main__":
    # init agent
    alpha = 1.2
    lamb = 0.95
    gamma = 1
    method = 'replacing'
    epsilon = 1
    num_of_tilings = 8
    max_size = 2048
    decay_agent_eps = None
    base_agent_class = SarsaLambdaCMAC3DMountainCar
    base_agent = SarsaLambdaCMAC3DMountainCar(alpha, lamb, gamma, method, epsilon, num_of_tilings, max_size)

    # experimental setup
    env_name = 'MountainCar3D-v0'
    env_max_steps = 5000
    rd_seed = 42
    max_episodes_per_trial = 1000
    num_trials = 1 
    update_agent = True
    start_learning_after = -1
    print_debug = True

    # whether to save sample transition data
    agent_info = SarsaLambdaAgentInfo(alpha, lamb, gamma, method, epsilon, num_of_tilings, max_size)
    experiment_info = ExperimentInfo(env_name, env_max_steps, 42, rd_seed, 'SarsaLambda')
    save_sample_data = True
    save_sample_every = 25
    sample_data_col_names = ['Trial', 'Episode', 'Step', 'Current_x_position', 'Current_x_velocity', 'Current_y_position', 'Current_y_velocity',
'Current_action', 'Reward', 'Next_x_position', 'Next_x_velocity', 'Next_y_position', 'Next_y_velocity', 'Next_action']
    sample_data_column_dtypes = ['int', 'int', 'int', 'float', 'float', 'float', 'float', 'int', 'int', 'float', 'float', 'float', 'float', 'int']
    sample_data_folder = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Evolutionary and Swarm Intelligence\\src\\GAME\\output\\11012022 Collect Samples 3DMC Full Explore\\"
    sample_data_filename = '3DMC_sample_data.csv'
    data_collector = RLSamplesCollector(experiment_info, agent_info, sample_data_col_names, sample_data_column_dtypes)

    # whether or not to save the agent's weights
    save_agent = False
    save_agent_every = 25
    save_agent_folder = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Evolutionary and Swarm Intelligence\\src\\GAME\\pickle\\11012022 Gen Learn Curves 3DMC\\"
    save_agent_filename = 'agent_alpha_{:.2f}_lamb_{:.2f}_gam_{:.2f}_eps_{:.2f}_method_{}_ntiles_{}_max_size_{}.pickle'.format(alpha, lamb, gamma, epsilon, method, num_of_tilings, max_size)

    # whether to evaluate the agent and save the evaluation data
    eval_agent = False
    save_eval_every = 25
    eval_data_col_names = ['Trial', 'Episode', 'Reward']
    eval_data_column_dtypes = ['int', 'int', 'int']
    save_eval_folder = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Evolutionary and Swarm Intelligence\\src\\GAME\\output\\11012022 Collect Samples 3DMC Full Explore\\"
    save_eval_filename = 'eval_3DMC_a{}_l{}_e{}_nt{}.csv'.format(alpha, lamb, epsilon, num_of_tilings)
    eval_data_collector = RLSamplesCollector(experiment_info, agent_info, eval_data_col_names, eval_data_column_dtypes)

    # run experiment
    average_steps_per_trial = MountainCar3DExperiment(base_agent_class, base_agent, decay_agent_eps, max_episodes_per_trial, 
    num_trials, update_agent, start_learning_after, print_debug, save_sample_data, save_sample_every, 
    sample_data_col_names, sample_data_folder, sample_data_filename, data_collector, save_agent, save_agent_every,
    save_agent_folder, save_agent_filename, eval_agent, save_eval_every, eval_data_col_names, save_eval_folder, save_eval_filename, 
    eval_data_collector, env_name, env_max_steps, rd_seed)