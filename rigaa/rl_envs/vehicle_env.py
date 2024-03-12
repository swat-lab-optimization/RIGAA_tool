"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for traning an RL agent for road topology generation
"""


from gym import Env
from gym.spaces import Box, MultiDiscrete
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time
import logging as log

from shapely.geometry import LineString
from descartes import PolygonPatch

import config as cf
from rigaa.utils.car_road import Map
from rigaa.utils.vehicle_evaluate import interpolate_road
from rigaa.utils.vehicle_evaluate import evaluate_scenario, is_valid_road
from freneticlib.executors.bicycle.carlapidonbicycle import execute_carla_pid_on_bicycle
from rigaa.utils.road_validity_check import min_radius, interpolate_test

from shapely import geometry, ops

matplotlib.use("Agg")

MAX_RADIUS_THRESHOLD = 130
MIN_RADIUS_THRESHOLD = 47


class CarEnv(Env):
    def __init__(self):
        self.max_number_of_points = 31
        self.action_space = MultiDiscrete(
            [
                3,
                int(cf.vehicle_env["max_len"] - cf.vehicle_env["min_len"]),
                int(cf.vehicle_env["max_angle"] - cf.vehicle_env["min_angle"]) / 2,
            ]
        )
        self.done = False

        self.evaluate = False

        self.min_fitness = 3.2 #1.7

        self.dist_explored = []
        self.angle_explored = []

        self.max_steps = 30
        self.steps = 0
        self.old_fitness = 0

        self.max_fitness = 5 #1.65 2

        self.fitness = 0
        self.ran_prob = 0.1  # 0.05 #0.1

        self.car_path = []
        self.road = []

        self.t = 0

        self.episode = 0
        self.results = []
        self.all_results = {}
        self.train_episode = 0
        self.evaluation = False
        self.scenario = []
        self.observation_space = Box(
            low=0, high=100, shape=(self.max_number_of_points * 3,), dtype=np.int8
        )

    def generate_init_state(self):

        self.state = np.zeros((self.max_number_of_points, 3))

        road_type = np.random.randint(0, 2)
        value = np.random.randint(
            cf.vehicle_env["min_len"], cf.vehicle_env["max_len"] + 1
        )
        position = np.random.randint(
            cf.vehicle_env["min_angle"], cf.vehicle_env["max_angle"] + 1
        )
        state = [road_type, value, position]
        road_type2 = np.random.randint(0, 2)
        value2 = np.random.randint(
            cf.vehicle_env["min_len"], cf.vehicle_env["max_len"] + 1
        )
        position2 = np.random.randint(
            cf.vehicle_env["min_angle"], cf.vehicle_env["max_angle"] + 1
        )
        state2 = [road_type2, value2, position2]

        road_type3 = np.random.randint(0, 2)
        value3 = np.random.randint(
            cf.vehicle_env["min_len"], cf.vehicle_env["max_len"] + 1
        )
        position3 = np.random.randint(
            cf.vehicle_env["min_angle"], cf.vehicle_env["max_angle"] + 1
        )

        state3 = [road_type3, value3, position3]




        self.angle_explored = [position, position2, position3]
        self.dist_explored = [value, value2, value2]

        self.state[0] = state
        self.state[1] = state2  # need two states to evaluate the initial fitness
        self.state[2] = state3

    def set_state(self, action):
        if action[0] == 0:
            distance = action[1] + cf.vehicle_env["min_len"]
            angle = 0
        elif action[0] == 1:
            angle = action[2] * 2 + cf.vehicle_env["min_angle"]
            distance = 0
        elif action[0] == 2:
            angle = action[2] * 2 + cf.vehicle_env["min_angle"]
            distance = 0

        return [action[0], distance, angle]
    

    def eval_fitness(self, states):
        points, _ = self.map.get_points_from_states(states)
        intp_points = interpolate_road(points)
        fitness, car_path = evaluate_scenario(intp_points, rl_train=True)

        return abs(fitness), car_path, intp_points
    

    def get_deviation(self, lane_center, car_points):

        deviations = []

        for p in car_points:
            p = geometry.Point(p[0], p[1])
            deviation = geometry.LineString(ops.nearest_points(lane_center, p)).length
            deviations.append(deviation)

        #print(deviations)

        return max(deviations)


    def eval_fitness_curve(self, states): # curve based


        points, _ = self.map.get_points_from_states(states)
        intp_points = interpolate_road(points)
        fitness = 0
        car_path = [[], []]


        test = interpolate_test(points)


        min_curve = min_radius(test)

        if min_curve <= MIN_RADIUS_THRESHOLD:
            fitness = 0
        else:
            fitness = 1/min_curve*100

        #log.info(f"Fitness: {fitness}")
        #fitness = 0 
        
        return  abs(fitness), car_path, intp_points


    def eval_fitness_simple(self, states):
        points, _ = self.map.get_points_from_states(states)
        intp_points = interpolate_road(points)
        fitness = 0
        car_path = [[], []]

        if is_valid_road(intp_points): 
            x_coordinates, y_coordinates = zip(*intp_points)

            #fitness, car_path = evaluate_scenario(intp_points, rl_train=True)
            data = execute_carla_pid_on_bicycle(x_coordinates, y_coordinates, desired_speed=70,  pid_gains_long={"K_P": 5, "K_D": 0.01, "K_I": 0.01})
            p_x = data["pxs"]
            p_y = data["pys"]
            car_path = [p_x, p_y]
            lane_center =  geometry.LineString(np.array(intp_points))
            fitness = self.get_deviation(lane_center, zip(p_x, p_y))


        return abs(fitness), car_path, intp_points


    def step_new(self, action):

        start = time.time()
        assert self.action_space.contains(action)
        self.done = False

        self.state[self.steps] = self.set_state(action)

        dist = self.state[self.steps][1]
        angle = self.state[self.steps][2]

        self.fitness, self.car_path, _ = self.eval_fitness_simple(self.state[: self.steps])

        improvement = self.fitness - self.old_fitness

        if self.fitness == 0:
            reward = -5#0
            self.done = True
        else:
            reward = self.fitness
            if self.fitness < self.min_fitness:
                reward = 0
            if improvement > 0:
                reward += improvement * 5  # 10 *
            if self.fitness > self.max_fitness:
                reward += 10

            if not (dist in self.dist_explored):
                reward += 1
                self.dist_explored.append(dist)

            if not (angle in self.angle_explored):
                reward += 1
                self.angle_explored.append(angle)

        self.old_fitness = self.fitness
        
        log.debug(f"Step time: {time.time() - start}")
        self.steps += 1

        if self.steps >= self.max_steps:
            self.done = True

        info = {}
        obs = [coordinate for tuple in self.state for coordinate in tuple]

        #self.render()

        return np.array(obs, dtype=np.int8), reward, self.done, info

    def step(self, action):

        start = time.time()
        assert self.action_space.contains(action)
        self.done = False

        self.state[self.steps] = self.set_state(action)

        dist = self.state[self.steps][1]
        angle = self.state[self.steps][2]

        self.fitness, self.car_path, _ = self.eval_fitnes(self.state[: self.steps])

        improvement = self.fitness - self.old_fitness

        if self.fitness < 0:
            reward = -50
            self.done = True
        else:
            reward = self.fitness
            if improvement > 0:
                reward += improvement * 2  # 10 *
            if self.fitness > self.max_fitness:
                reward += 50

            if not (dist in self.dist_explored):
                reward += 1
                self.dist_explored.append(dist)

            if not (angle in self.angle_explored):
                reward += 1
                self.angle_explored.append(angle)

        self.old_fitness = self.fitness
        
        log.debug(f"Step time: {time.time() - start}")
        self.steps += 1

        if self.steps >= self.max_steps:
            self.done = True

        info = {}
        obs = [coordinate for tuple in self.state for coordinate in tuple]



        return np.array(obs, dtype=np.int8), reward, self.done, info

    def reset(self):
        # print("Reset")



        self.map = Map(cf.vehicle_env["map_size"])

        self.steps = 3
        # print(self.fitness)
        self.generate_init_state()  # generate_random_state()#road()

        self.old_fitness, _, _ = self.eval_fitness_simple(self.state[: self.steps])

        obs = [coordinate for tuple in self.state for coordinate in tuple]

        return np.array(obs, dtype=np.int8)

    def get_size(self, states):
        size = 0
        for state in states:
            if state != [0, 0, 0]:
                size += 1
        return size

    def render(self, scenario=[], img_path="./", mode="human"):

        scenario = self.state[: self.steps]

        # log.info(f"Scenario {scenario}")

        fitness, car_path, intp_points = self.eval_fitness(scenario)
        # if self.done:
        fig, ax = plt.subplots(figsize=(12, 12))
        road_x = []
        road_y = []
        for p in intp_points:
            road_x.append(p[0])
            road_y.append(p[1])

        ax.plot(road_x, road_y, "yo--", label="Road")

        if len(car_path):
            ax.plot(car_path[0], car_path[1], "bo", label="Car path")

        road_poly = LineString([(t[0], t[1]) for t in intp_points]).buffer(
            8.0, cap_style=2, join_style=2
        )
        road_patch = PolygonPatch(
            (road_poly), fc="gray", ec="dimgray"
        )  # ec='#555555', alpha=0.5, zorder=4)
        ax.add_patch(road_patch)

        top = cf.vehicle_env["map_size"]
        bottom = 0
        ax.set_title("Test case fitenss " + str(fitness), fontsize=17)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.legend(fontsize=16)
        ax.set_ylim(bottom, top)
        plt.ioff()
        ax.set_xlim(bottom, top)
        ax.legend()

        os.makedirs(img_path, exist_ok=True)

        if self.evaluate:
            fig.savefig(
                f"{img_path}{self.episode}_{fitness:.2f}.png", bbox_inches="tight"
            )
        else:
            os.makedirs("debug", exist_ok=True)
            fig.savefig("debug\\debug_step_" + str(self.steps) + ".png")

        plt.close(fig)


class CarEnvEval(CarEnv):

    def __init__(self):
        super().__init__()

    def step(self, action):
        assert self.action_space.contains(action)
        self.done = False

        self.state[self.steps] = self.set_state(action)

        reward = 0

        self.steps += 1

        if self.steps >= self.max_steps:

            self.done = True

        info = {}
        obs = [coordinate for tuple in self.state for coordinate in tuple]

        return np.array(obs, dtype=np.int8), reward, self.done, info

    def reset(self):
        # print("Reset")

        self.map = Map(cf.vehicle_env["map_size"])

        self.steps = 2
        # print(self.fitness)
        self.generate_init_state()  # generate_random_state()#road()

        obs = [coordinate for tuple in self.state for coordinate in tuple]

        return np.array(obs, dtype=np.int8)
