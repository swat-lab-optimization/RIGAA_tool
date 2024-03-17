"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for saving test scenario metadata such as the test suite statistics and the test suite itself
"""

import os
import json
import logging as log
from datetime import datetime
import config as cf


def save_tc_results(tc_stats, tcs, tcs_convergence, tcs_hyper, algo, problem, ro):
    """
    It takes two arguments, tc_stats and tcs, and saves them as JSON files in the directories specified
    in the config file

    Args:
      tc_stats: a dictionary of the test cases statistics
      tcs: a list of test cases
    """

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y")

    stats_path = dt_string + "_" + cf.files["stats_path"] + "_" + algo + "_" + problem + "_" + ro
    tcs_path = dt_string + "_" + cf.files["tcs_path"] + "_" + algo + "_" + problem + "_" + ro

    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
    if not os.path.exists(tcs_path):
        os.makedirs(tcs_path)

    with open(
        os.path.join(stats_path, dt_string + "-stats.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(tc_stats, f, indent=4)
        log.info(
            "Stats saved as %s", os.path.join(stats_path, dt_string + "-stats.json")
        )

    with open(
        os.path.join(stats_path, dt_string + "-conv.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(tcs_convergence, f, indent=4)
        log.info(
            "Stats saved as %s", os.path.join(stats_path, dt_string + "-conv.json")
        )

    with open(
        os.path.join(stats_path, dt_string + "-hyper.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(tcs_hyper, f, indent=4)
        log.info(
            "Stats saved as %s", os.path.join(stats_path, dt_string + "-hyper.json")
        )

    with open(
        os.path.join(tcs_path, dt_string + "-tcs.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(tcs, f, indent=4)
        log.info(
            "Test cases saved as %s", os.path.join(tcs_path, dt_string + "-tcs.json")
        )
