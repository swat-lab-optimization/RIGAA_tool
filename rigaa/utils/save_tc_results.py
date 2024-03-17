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

from simulator.code_pipeline.tests_evaluation import OOBAnalyzer
from simulator.code_pipeline.tests_generation import TestGenerationStatistic
def save_tc_results(tc_stats, tcs, tcs_convergence, tcs_hyper, all_tests, dt_string, algo, problem, name):
    """
    It takes two arguments, tc_stats and tcs, and saves them as JSON files in the directories specified
    in the config file

    Args:
      tc_stats: a dictionary of the test cases statistics
      tcs: a list of test cases
    """


    stats_path = dt_string + "_" + cf.files["stats_path"] + "_" + algo + "_" + problem + name
    tcs_path = dt_string + "_" + cf.files["tcs_path"] + "_" + algo + "_" + problem + name

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
    if len(all_tests) > 0:
        with open(
            os.path.join(stats_path, dt_string + "-all_tests.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(all_tests, f, indent=4)
            log.info(
                "Stats saved as %s", os.path.join(stats_path, dt_string + "-all_tests.json")
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


def create_summary(result_folder, raw_data):
    log.info(f"Creating Reports based on {result_folder}")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Refactor this
    if type(raw_data) is TestGenerationStatistic:
        log.info("Creating Test Statistics Report:")
        summary_file = os.path.join(result_folder, "generation_stats.csv")
        csv_content = raw_data.as_csv()
        with open(summary_file, 'w') as output_file:
            output_file.write(csv_content)
        log.info("Test Statistics Report available: %s", summary_file)

    log.info("Creating OOB Report")
    oobAnalyzer = OOBAnalyzer(result_folder)
    oob_summary_file = os.path.join(result_folder, "oob_stats.csv")
    csv_content = oobAnalyzer.create_summary()
    with open(oob_summary_file, 'w') as output_file:
        output_file.write(csv_content)

    log.info("OOB  Report available: %s", oob_summary_file)