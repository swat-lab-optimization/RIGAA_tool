"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for evaluating the diversity of test scenarios
"""

from rigaa.solutions import VehicleSolution
from rigaa.solutions import RobotSolution


def calc_novelty(state1, state2, problem):
    """
    > The function takes two states and a problem type as input and returns the novelty of the two
    states

    Args:
      state1: the first state to compare
      state2: the state to compare to
      problem: the problem we're solving, either "vehicle" or "robot"

    Returns:
      The novelty of the solution relative to the other solutions in the test suite.
    """

    if problem == "vehicle":
        novelty = abs(VehicleSolution().calculate_novelty(state1, state2))
    elif problem == "robot":
        novelty = abs(RobotSolution().calculate_novelty(state1, state2))

    return novelty
