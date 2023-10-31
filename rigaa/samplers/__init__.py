
from rigaa.samplers.robot_sampling import RobotSampling
from rigaa.samplers.vehicle_sampling import VehicleSampling


from rigaa.rl_agents.vehicle_agent import generate_rl_road
from rigaa.samplers.vehicle_sampling import generate_random_road
from rigaa.samplers.vehicle_sampling import generate_frenetic_road
from rigaa.rl_agents.robot_agent import generate_rl_map
from rigaa.samplers.robot_sampling import generate_random_solution


SAMPLERS = {
    "vehicle": VehicleSampling,
    "robot": RobotSampling,
}

GENERATORS ={
    "vehicle": generate_random_road,
    "vehicle_rl": generate_rl_road,
    "vehicle_frenetic": generate_frenetic_road,
    "robot": generate_random_solution,
    "robot_rl": generate_rl_map
}