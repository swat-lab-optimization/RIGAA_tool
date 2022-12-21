

from gym import Env
from gym.spaces import Box, MultiDiscrete
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import math
import os
import config as cf
from rigaa.utils.robot_map import Map
from rigaa.utils.a_star import AStarPlanner

class RobotEnv(Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        self.max_number_of_points = cf.robot_env["map_size"] - 2 
        self.action_space = MultiDiscrete([2, cf.robot_env['max_len'] - cf.robot_env['min_len'], cf.robot_env['max_pos'] - cf.robot_env['min_pos']])  # 0 - increase temperature, 1 - decrease temperature
        self.observation_space = Box(low=0, high=self.max_number_of_points, shape=(self.max_number_of_points*3,), dtype=np.int8)
        #self.observation_space = Box(low=0, high=255, shape=(cf.robot_env["map_size"], cf.robot_env["map_size"], 1), dtype=np.uint8)

        self.sx = 1.0  # [m]
        self.sy = 1.0  # [m]
        self.gx = cf.robot_env["map_size"] - 2  # [m]
        self.gy = cf.robot_env["map_size"] - 2 # [m]
        #self.map_builder = Map(cf.robot_env["map_size"])
        self.bonus = 0
        self.all_states = []
        self.all_fitness = []

        self.grid_size = 1  # [m]
        self.robot_radius = 0.1  # [m]
       
        self.episode = 0
        self.max_steps = 40   
        self.evaluation = False
        self.fitness = 0


    def generate_init_state(self):
        self.state = np.zeros((self.max_number_of_points, 3))
        random_position = 0

        ob_type = np.random.randint(0, 2)
        value = np.random.randint(cf.robot_env["min_len"], cf.robot_env["max_len"] + 1)
        position = np.random.randint(cf.robot_env["min_pos"], cf.robot_env["max_pos"] + 1)
        self.state[random_position]  = np.array([ob_type, value, position])
        self.old_location = position
        self.position_explored = [[ob_type, position]]
        self.sizes_explored = [value]

    def get_length(self, rx, ry):
        total_len = 0
        
        for i in range(1, len(rx)):
            len_ = math.sqrt((rx[i] - rx[i-1])**2 + (ry[i] - ry[i-1])**2)
            total_len += len_

        return total_len



    def eval_fitness(self, map_points):
        ox = [t[0] for t in map_points]
        oy = [t[1] for t in map_points] 


        a_star = AStarPlanner(ox, oy, self.grid_size, self.robot_radius)  # noqa: E501

        rx, ry, _ = a_star.planning(self.sx, self.sy, self.gx, self.gy)

        path = zip(rx, ry)
        
        scenario_size  = self.get_size()
        if len(rx) > 2:
        
            test_road = LineString([(t[0], t[1]) for t in path])
            self.fitness = test_road.length
        else:
            self.fitness = -10

            self.done = True


        return self.fitness

    def get_size(self):
        num = 0
        for i in (self.state):
            if i[1] != 0:
                num += 1
        return num
                
 
    def step(self, action):
        assert self.action_space.contains(action)
        self.done = False

        self.state[self.steps] = self.set_state(action)   
        reward = 0

        if self.steps >= self.max_steps - 3:
            self.done = True
            map_builder = Map(cf.robot_env["map_size"])
            map_points = map_builder.get_points_from_states(self.state)

            points_list = map_builder.get_points_cords(map_points)

            new_reward = self.eval_fitness(points_list)


        '''

        map_builder = Map(cf.robot_env["map_size"])
        map_points = map_builder.get_points_from_states(self.state)

        points_list = map_builder.get_points_cords(map_points)

        new_reward = self.eval_fitness(points_list) # - discount
        current_state = self.state.copy()

        improvement = new_reward - self.old_reward
        position = [action[0], action[2] + cf.robot_env['min_pos']]
        value = action[1] + cf.robot_env['min_len']
        

        if new_reward < 0:
            reward = -10
        else:

            reward = new_reward - self.init_fitness

            if improvement > 0:
                reward += improvement*10

            if not(value in self.sizes_explored):
                reward += 5
                self.sizes_explored.append(value)

            if not(position in self.position_explored):
                reward += 5
                self.position_explored.append(position)
        

        self.old_reward = new_reward
        
        self.all_fitness.append(self.fitness)
        self.all_states.append(current_state)
        '''

        self.steps += 1

        info = {}

        obs = [coordinate for tuple in self.state for coordinate in tuple]

        

        return np.array(obs, dtype=np.int8), reward, self.done, info

    def reset(self):

        #print(self.fitness)

        self.generate_init_state()

        map_builder = Map(cf.robot_env["map_size"])
        map_points = map_builder.get_points_from_states(self.state)
        points_list = map_builder.get_points_cords(map_points)

        self.scenario_size = self.get_size() 

        default_reward = 0#self.eval_fitness(points_list)

        
        self.old_reward = default_reward#bonus

        self.init_fitness = default_reward



        self.all_states = []
        self.all_fitness = []

        self.fitness = 0

        self.steps = 1

        self.done = False

        obs = [coordinate for tuple in self.state for coordinate in tuple]

        return np.array(obs, dtype=np.int8)

    def render(self, scenario):

        #if self.done:

        fig, ax = plt.subplots(figsize=(12, 12))

        map_builder = Map(cf.robot_env["map_size"])
        map_points = map_builder.get_points_from_states(scenario)
        points_list = map_builder.get_points_cords(map_points)



        road_x = []
        road_y = []
        for p in points_list:
            road_x.append(p[0])
            road_y.append(p[1])


        a_star = AStarPlanner(road_x, road_y, self.grid_size, self.robot_radius)  # noqa: E501

        rx, ry, _ = a_star.planning(self.sx, self.sy, self.gx, self.gy)

        path = zip(rx, ry)


        test_road = LineString([(t[0], t[1]) for t in path])
        fit = test_road.length

  

        ax.plot(rx, ry, '-r', label="Robot path")

        ax.scatter(road_x, road_y, s=150, marker='s', color='k', label="Walls")

        top = cf.robot_env["map_size"]
        bottom = 0

        ax.tick_params(axis='both', which='major', labelsize=18)

        ax.set_ylim(bottom, top)
        
        ax.set_xlim(bottom, top)
        ax.legend(fontsize=22)

        top = cf.robot_env["map_size"] + 1
        bottom = - 1


        #if os.path.exists(cf.files["img_path"]) == False:
        #        os.mkdir(cf.files["img_path"])

        #fig.savefig(cf.files["img_path"] + str(self.episode) + "_" + str(fit) + ".png")

        fig.savefig("test.png")

        plt.close(fig)



    def set_state(self, action):
        #if action[0] == 0:
        obs_size = action[1] + cf.robot_env['min_len']
        position = action[2] + cf.robot_env['min_pos']

        return [action[0], obs_size, position]