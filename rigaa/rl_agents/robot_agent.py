from stable_baselines3 import PPO
from rigaa.rl_envs.robot_env import RobotEnv

import numpy as np
def generate_rl_map():
    #model_save_path = "models//rl_model_09-21_mlp_v0_1_120000_steps.zip"
    #model_save_path = "models//rl_model_09-24_mlp_v0_0_500000_steps.zip"
    #model_save_path = "models\\2023-01-21-rl_model-0_540000_steps.zip"
    #model_save_path = "models\\2023-01-28-rl_model-0_600000_steps.zip"
    model_save_path = "models/2023-01-28-rl_model-0_600000_steps.zip"
    model = PPO.load(model_save_path)

    environ = RobotEnv()

    scenario_found = False
    #scenario_list = []

    i = 0
    while scenario_found == False:
        obs = environ.reset()
        done = False
        #i = 0
        while not done:
            action, _ = model.predict(obs)

            obs, rewards, done, info = environ.step(action)
        i += 1
        fitness = environ.fitness

        #print(max_fitness)

        if (fitness > 70) or i >15:
            #print(i)
            #print("Round: {}".format(environ.episode))
            #print("Max fitness: {}".format(max_fitness))
            
            #scenario = environ.state[:environ.t]
            scenario = environ.state

            #environ.render(scenario)
            #scenario_list.append(scenario)
            scenario_found = True
            print("Found scenario after %d attempts" % i)

            i  = 0


    return scenario, -fitness