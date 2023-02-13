
#ga = {"pop_size": 200, "n_gen": 50,  "mut_rate": 0.4, "cross_rate": 0.9, "test_suite_size": 30} #robot
ga = {"pop_size": 150, "n_gen": 50,  "mut_rate": 0.4, "cross_rate": 0.9, "test_suite_size": 30} #vehice
files = {"stats_path": "stats", "tcs_path": "tcs", "images_path": "tc_images"}
rl = {"init_pop_prob": 0.2}

vehicle_env = {
    "map_size": 200,
    "min_len": 5,  # minimal possible distance in meters
    "max_len": 30,  # maximal possible disance to go straight in meters
    "min_angle": 10,  # minimal angle of rotation in degrees
    "max_angle": 80,  # maximal angle of rotation in degrees
    "speed": 9,
    "steer_ang": 12,
}

robot_env = {
    "map_size": 40,
    "min_len": 8,  # 3,  # minimal possible distance in meters
    "max_len": 15,  # 12,#15,  # maximal possible disance to go straight in meters
    "min_pos": 1,  # minimal possible
    "max_pos": 38,  # 49 29,  # maximal possible position in meters
    "grid_size": 1,  # 1,  # size of the grid in meters
    "robot_radius": 0.1,  # 1,  # radius of the robot in meters
    "start": 1,  # 1,  # start x position
    "goal": 38,  # 1,  # start y position
}



