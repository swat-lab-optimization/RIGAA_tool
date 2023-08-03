
from stable_baselines3 import PPO
import numpy as np
import logging as log

from rigaa.rl_envs.vehicle_env import CarEnvEval
from rigaa.utils.vehicle_evaluate import interpolate_road
from rigaa.utils.vehicle_evaluate import evaluate_scenario
from rigaa.utils.car_road import Map
import config as cf

#from rigaa.samplers.vehicle_sampling import generate_random_road

def generate_rl_road():
    #model_save_path = "models\\2023-01-18-rl_model-0_20000_steps.zip"
    #model_save_path = "models//rl_model_2022-10-070_2790000_steps.zip"
    #model_save_path = "models\\2023-01-19-rl_model0_3000000_steps.zip"
    #model_save_path  = "models\\2023-01-19-rl_model30_2140000_steps.zip"
    #model_save_path = "models\\2023-01-24-rl_model_nt_koef_0010_3000000_steps.zip"
    model_save_path  = "models\\2023-01-26-rl_model_nt_koef_0010_2740000_steps.zip"
    model = PPO.load(model_save_path)

    environ = CarEnvEval()

    scenario_found = False
    i = 0
    while scenario_found == False:
        obs = environ.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs)

            obs, rewards, done, info = environ.step(action)
        i += 1
        #car = Car(cf.vehicle_env["speed"], cf.vehicle_env["steer_ang"], cf.vehicle_env["map_size"])
        map = Map(cf.vehicle_env["map_size"])
        scenario = environ.all_states[-1]
        points, scenario = map.get_points_from_states(scenario)
        intp_points = interpolate_road(points)

        max_fitness, _ = (evaluate_scenario(intp_points))
        max_fitness = abs(max_fitness)

        if (max_fitness > 3.2) or i > 10:

            scenario_found = True
            #if i > 10:
            #    scenario = generate_random_road()

            log.debug(f"Found scenario after {i} attempts")

            i  = 0
            


    return scenario, -max_fitness