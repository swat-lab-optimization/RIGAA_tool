
from stable_baselines3 import PPO
import numpy as np

from rigaa.rl_envs.vehicle_env import CarEnv
from rigaa.utils.vehicle import Car
from rigaa.utils.car_road import Map
import config as cf

def generate_rl_road():
    model_save_path = "models//rl_model_2022-10-070_2790000_steps.zip"
    model = PPO.load(model_save_path)

    environ = CarEnv()

    scenario_found = False

    while scenario_found == False:
        obs = environ.reset()
        done = False
        i = 0
        while not done:
            action, _ = model.predict(obs)

            obs, rewards, done, info = environ.step(action)
        i += 1
        car = Car(cf.vehicle_env["speed"], cf.vehicle_env["steer_ang"], cf.vehicle_env["map_size"])
        map = Map(cf.vehicle_env["map_size"])
        scenario = environ.all_states[-2]
        points = map.get_points_from_states(scenario)
        intp_points = car.interpolate_road(points)

        max_fitness, _ = (car.execute_road(intp_points))
        max_fitness = abs(max_fitness)

        if (max_fitness > 3.7) or i > 15:

            scenario_found = True

            i  = 0


    return scenario, -max_fitness