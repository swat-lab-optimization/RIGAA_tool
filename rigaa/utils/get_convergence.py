"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for getting the convergence of the search algoritm (best values at each generation)
"""

import numpy as np
import config as cf
from pymoo.indicators.hv import HV

def get_convergence(res, n_offsprings):
    """
    It takes the result of the genetic algorithm and returns a list of the best fitness values of each generation.

    Args:
      res: the result of the genetic algorithm
    """
    res_dict = {}
    hyper_dict = {}
    generations = np.arange(0, len(res.history), 1)
    convergence = []
    hyper_volume = []
    if res.problem._name == "vehicle":
        ref_point = np.array([16, 1])
    elif res.problem._name == "robot":
        ref_point = np.array([280, 1])
    else:
        ref_point = np.array([1, 1])
   
    for gen in generations:
        population = -res.history[gen].pop.get("F")
        population = sorted(population, key=lambda x: x[0], reverse=True)
        convergence.append(population[0][0])

        opt_num = len(res.history[gen].opt)
        pareto = res.history[gen].pop.get("F")[:opt_num]*(-1)
        
        hv = HV(ref_point=ref_point) 
        print(hv(pareto))
        hyper_volume.append(hv(pareto))

    step = n_offsprings
    evaluations = np.arange(
        cf.ga["pop_size"], len(res.history) * n_offsprings + cf.ga["pop_size"], step
    )

    for i in range(len(evaluations)):
        res_dict[str(evaluations[i])] = convergence[i]
    for i in range(len(evaluations)):
        hyper_dict[str(evaluations[i])] = hyper_volume[i]
    return res_dict, hyper_dict
