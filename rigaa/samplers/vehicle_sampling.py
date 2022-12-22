
import logging as log
import numpy as np
from pymoo.core.sampling import Sampling
import config as cf
import time
from rigaa.utils.car_road import Map
from rigaa.utils.vehicle import Car
from rigaa.solutions import VehicleSolution
from rigaa.rl_agents.vehicle_agent import generate_rl_road

def generate_random_road():
    """
    It generates a random road topology
    """
    actions = list(range(0, 3))
    lengths = list(range(cf.vehicle_env["min_len"], cf.vehicle_env["max_len"]))
    angles = list(range(cf.vehicle_env["min_angle"], cf.vehicle_env["max_angle"]))

    map_size = cf.vehicle_env["map_size"]

    speed = cf.vehicle_env["speed"]
    steer_ang = cf.vehicle_env["steer_ang"]

    fitness = 0

    while fitness == 0:  # ensures that the generated road is valid
        done = False
        test_map = Map(map_size)
        car = Car(speed, steer_ang, map_size)
        while not done:
            action = np.random.choice(actions)
            if action == 0:
                length = np.random.choice(lengths)
                done = not (test_map.go_straight(length))
            elif action == 1:
                angle = np.random.choice(angles)
                done = not (test_map.turn_right(angle))
            elif action == 2:
                angle = np.random.choice(angles)
                done = not (test_map.turn_left(angle))
        scenario = test_map.scenario[:-1]

        road_points = test_map.get_points_from_states(scenario)
        intp_points = car.interpolate_road(road_points)
        fitness, _ = car.execute_road(intp_points)

    return scenario, fitness


class VehicleSampling(Sampling):

    """
    Module to generate the initial population

    returns: a tensor of candidate solutions
    """

    def __init__(self, init_pop_prob):
        super().__init__()
        self.init_pop_prob = init_pop_prob
    def _do(self, problem, n_samples, **kwargs):

        X = np.full((n_samples, 1), None, dtype=object)

        for i in range(n_samples):
            r = np.random.random()
            if r < self.init_pop_prob:
                start = time.time()
                states, fitness = generate_rl_road()
                log.debug("Individual produced by RL in %f sec", time.time() - start)
            else:
                start = time.time()
                states, fitness = generate_random_road()
                log.debug("Individual produced by randomly in %f sec", time.time() - start)
            s = VehicleSolution()
            s.states = states
            s.fitness = fitness
            X[i, 0] = s

        log.debug("Initial population of %d solutions generated", n_samples)
        return X
