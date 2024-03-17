
ga = {"pop_size": 40, "n_gen": 35,  "mut_rate": 0.4, "cross_rate": 0.9, "test_suite_size": 40} # specify the population size here
files = {"stats_path": "stats", "tcs_path": "tcs", "images_path": "tc_images"}
rl = {"init_pop_prob": 0.4}

vehicle_env = {
    "map_size": 200,
    "min_len": 5,  # minimal possible straight road segment length 
    "max_len": 30,  # maximal possible traight road segment length 
    "min_angle": 10,  # minimal angle of rotation in degrees
    "max_angle": 80  # maximal angle of rotation in degrees
}

robot_env = {
    "map_size": 40,
    "min_len": 8,  # minimal obstacle size
    "max_len": 15,  # max obstacle size
    "min_pos": 1,  # minimal obscacle position on the x axis
    "max_pos": 38,  # maximal obscacle position on the x axis
    "grid_size": 1, # A* algoritrhm hyperparameter
    "robot_radius": 0.1,  # A* algoritrhm hyperparameter
    "sx": 1, # start x position
    "sy": 1, # start y position, when using Ant maze simulatior change to 38
    "gx": 38, # goal x position, when using Ant maze simulatior change to 36
    "gy": 38 # goal y position, when using Ant maze simulatior change to 3
    }



