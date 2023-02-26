
import numpy as np
from rigaa.utils.robot_map import Map
import config as cf
from rigaa.utils.rgp import rdp
from rigaa.utils.a_star import AStarPlanner
#from rigaa.solutions.robot_solution import RobotSolution

def ant_adapt_points(points):
    points = points[::-1]
    max_size = cf.robot_env["map_size"] - 1
    new_points = [[p[0], max_size - p[1]] for p in points]
    return new_points

def get_d4rl_map(states):
    map_builder = Map(cf.robot_env["map_size"])
    map_points = map_builder.get_points_from_states(states, full=True)
    points_list = map_builder.get_points_cords(map_points)
    ant_map_points = np.logical_not(map_points).astype(int)
    ant_map_points = [list(row) for row in ant_map_points]

    grid_size = cf.robot_env["grid_size"]  # [m]
    robot_radius = cf.robot_env["robot_radius"]  # [m]

    ox = [t[0] for t in points_list]
    oy = [t[1] for t in points_list]
    sx = cf.robot_env["sx"]  # [m]
    sy = cf.robot_env["sy"]  # [m]
    gx = cf.robot_env["gx"]  # [m]
    gy = cf.robot_env["gy"]  # [m]

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)  # noqa: E501
    rx, ry, rtime = a_star.planning(sx, sy, gx, gy)
    print("Time taken", rtime)
    #print("original points", list(zip(rx, ry)))
    points = rdp(list(zip(rx, ry)), 0.9)
    print("Approximated points", (points[::-1]))
    adopt_points = ant_adapt_points(points)
    print("Adopted points", adopt_points)



    return ant_map_points, adopt_points
    