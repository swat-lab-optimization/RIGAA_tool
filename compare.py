"""
Author: Dmytro Humeniuk, SWAT Lab, Polytechnique Montreal
Date: 2023-08-10
Description: script for processing the experimental results
"""

import os
import argparse
import csv
from itertools import combinations
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import logging as log
from scipy.stats import mannwhitneyu

from matplotlib.ticker import MaxNLocator

from rigaa.utils.cliffsDelta import cliffsDelta
from rigaa.utils.calc_novelty import calc_novelty

def setup_logging(log_to, debug):
    """
    It sets up the logging system
    """

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, "w", "utf-8")
        log_handlers.append(file_handler)
        start_msg += " ".join([", writing logs to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=log_level,
        handlers=log_handlers,
        force=True,
    )
    log.info(start_msg)

def parse_arguments():
    """
    This function parses the arguments passed to the script
    :return: The arguments that are being passed to the program
    """

    log.info("Parsing the arguments")
    parser = argparse.ArgumentParser(
        prog="compare.py",
        description="A tool for generating test cases for autonomous systems",
        epilog="For more information, please visit ",
    )
    parser.add_argument(
        "--stats_path",
        nargs="+",
        help="The source folders of the metadate to analyze",
        required=True,
    )
    parser.add_argument(
        "--stats_names",
        nargs="+",
        help="The names of the corresponding algorithms",
        required=True,
    )
    parser.add_argument(
        "--plot_name", help="Name to add to the plots", required=False, default=""
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Argument to parse the results from tools evaluation",
    )
    parser.add_argument(
        "--all_tests",
        action="store_true",
        help="Argument to parse the results from tools evaluation",
    )
    in_arguments = parser.parse_args()
    return in_arguments


def compare_mean_best_values_found(best_fitness_list, column_names, plot_name):
    """
    This function compares the mean best values found for a given problem and writes the results to a
    CSV file.

    Args:
      best_fitness_list: A list of lists containing the best fitness values found for each run of an
    optimization algorithm.
      column_names: A list of strings representing the names of the columns in the output CSV file
      problem: The name
      plot_name: The name of the plot or file that will be created.
    """
    if "vehicle" in plot_name:
        problem = "vehicle"
    elif "robot" in plot_name:
        problem = "robot"
    else:
        print("Invalid plot name. Add vehicle or robot")
        sys.exit()

    columns = ["Problem", "Metric"]
    for name in column_names:
        columns.append(name)

    row_0 = [problem, "Mean best value found"]
    for val in best_fitness_list:
        row_0.append(round(np.mean(val), 3))

    rows = [columns, row_0]
    log.info("Writing mean best values to " + plot_name + "_mean_best_val.csv")
    with open(plot_name + "_mean_best_val.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


def compare_p_val_best_values_found(best_fitness_list, column_names, plot_name):
    """
    This function compares the p-values and effect sizes of the best fitness values found for different
    pairs of columns in a given problem and writes the results to a CSV file.

    Args:
      best_fitness_list: A list of lists containing the best fitness values found for each column in the
    dataset.
      column_names: A list of column names for the data being analyzed.
      problem: The problem being solved by the fitness function.
      plot_name: The name of the plot or analysis being performed, which will be used to name the output
    file.
    """
    title = ["A", "B", "p-value", "Effect size"]
    rows = [title]
    for pair in combinations(range(0, len(best_fitness_list)), 2):
        pair_values = []
        pair_values.append(column_names[pair[0]])
        pair_values.append(column_names[pair[1]])
        pair_values.append(
            mannwhitneyu(
                best_fitness_list[pair[0]],
                best_fitness_list[pair[1]],
                alternative="two-sided",
            )[1]
        )
        delta_value = round(
            cliffsDelta(best_fitness_list[pair[0]], best_fitness_list[pair[1]])[0], 3
        )
        delta_name = cliffsDelta(
            best_fitness_list[pair[0]], best_fitness_list[pair[1]]
        )[1]
        pair_values.append(str(delta_value) + str(", ") + delta_name)
        for i, p in enumerate(pair_values):
            if not (isinstance(p, str)):
                pair_values[i] = round(pair_values[i], 3)
        rows.append(pair_values)

    log.info(f"Writing p-values and effect sizes to {plot_name}_p_val_best_val.csv")
    with open(plot_name + "_res_p_best_val.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


def vargha_delaney_a12(x, y):
    """Calculate the Vargha-Delaney A12 effect size for continuous data.
    - x and y are arrays of continuous data.
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    mad_x = np.median(np.abs(x - np.median(x)))
    mad_y = np.median(np.abs(y - np.median(y)))
    mad = (mad_x * nx + mad_y * ny) / (nx + ny)
    a12 = (np.median(x) - np.median(y)) / mad
    return a12


def build_times_table(times_list, column_names):
    """
    The function `build_times_table` takes a list of times and column names, calculates mean generation
    time, p-value, and effect size, and writes the results to a CSV file.
    
    Args:
      times_list: times_list is a list of lists containing the generation times for different
    algorithms. Each inner list represents the generation times for a specific algorithm.
      column_names: The `column_names` parameter is a list of names for the columns in the times table.
    """
    columns = ["Metric"]
    for name in column_names:
        columns.append(name)
    columns.append("p-value")
    columns.append("Effect size")

    row_0 = ["mean generation time, s"]
    for alg in times_list:
        row_0.append(round(np.mean(alg), 3))
    row_0.append(
        round(mannwhitneyu(times_list[1], times_list[0], alternative="two-sided")[1], 3)
    )
    row_0.append(
        str(round(cliffsDelta(times_list[1], times_list[0])[0], 3))
        + ", "
        + str(cliffsDelta(times_list[1], times_list[0])[1])
    )

    row_1 = ["generation time std, s"]
    for alg in times_list:
        row_1.append(np.std(alg))

    rows = [columns, row_0]

    log.info(f"Writing the results of generation time to results_time.csv")
    with open("results_time.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


def build_median_table(fitness_list, diversity_list, column_names, plot_name):
    """
    The function `build_median_table` takes in fitness and diversity lists, column names, and a plot
    name, and creates a table with mean fitness and diversity values, as well as p-values and effect
    sizes if the lists have two elements each, and writes the table to a CSV file.
    
    Args:
      fitness_list: The `fitness_list` parameter is a list of lists, where each inner list represents
    the fitness values for a particular algorithm. Each inner list should contain the fitness values for
    a specific algorithm.
      diversity_list: The `diversity_list` parameter is a list of lists, where each inner list
    represents the diversity values for a specific algorithm. Each inner list should contain the
    diversity values for that algorithm.
      column_names: The parameter "column_names" is a list of names for the columns in the table. It is
    used to label the different algorithms or methods being compared in the table.
      plot_name: The `plot_name` parameter is a string that represents the name of the plot or table
    that will be generated. It will be used to create a CSV file with the results of the function.
    """
    columns = ["Metric"]
    for name in column_names:
        columns.append(name)

    row_0 = ["Mean fitness"]
    for alg in fitness_list:
        row_0.append(round(np.mean(alg), 3))

    row_1 = ["Mean diversity"]
    for alg in diversity_list:
        row_1.append(round(np.mean(alg), 3))

    if (len(fitness_list) == 2) and (len(diversity_list) == 2):
        row_0.append(
            round(
                mannwhitneyu(fitness_list[1], fitness_list[0], alternative="two-sided")[
                    1
                ],
                3,
            )
        )
        row_0.append(round(cliffsDelta(fitness_list[1], fitness_list[0])[0], 3))
        row_0.append(cliffsDelta(fitness_list[1], fitness_list[0])[1])

        row_1.append(
            round(
                mannwhitneyu(
                    diversity_list[0], diversity_list[1], alternative="two-sided"
                )[1],
                3,
            )
        )
        row_1.append(round(cliffsDelta(diversity_list[0], diversity_list[1])[0], 3))
        row_1.append(cliffsDelta(diversity_list[0], diversity_list[1])[1])
        columns.append("p-value")
        columns.append("Effect size")

    rows = [columns, row_0, row_1]

    log.info(f"Writing results to {plot_name}_res.csv")
    with open(plot_name + "_res.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


def build_cliff_data(fitness_list, diversity_list, column_names, plot_name):
    """
    The function `build_cliff_data` takes in fitness and diversity lists, column names, and a plot name,
    and writes the calculated p-values and effect sizes to separate CSV files for fitness and diversity.
    
    Args:
      fitness_list: The `fitness_list` parameter is a list of fitness values for different pairs of
    data. Each element in the list represents the fitness values for a specific pair of data.
      diversity_list: The `diversity_list` parameter is a list of lists, where each inner list
    represents the diversity values for a specific column or feature. Each inner list should contain the
    diversity values for that column or feature.
      column_names: The `column_names` parameter is a list of column names. It represents the names of
    the columns in the data that you want to analyze.
      plot_name: The `plot_name` parameter is a string that represents the name of the plot or data file
    that will be generated. It is used to create the output file names by appending
    "_res_p_value_fitness.csv" and "_res_p_value_diversity.csv" respectively.
    """
    title = ["A", "B", "p-value", "Effect size"]
    rows = [title]
    for pair in combinations(range(0, len(fitness_list)), 2):
        pair_values = []
        pair_values.append(column_names[pair[0]])
        pair_values.append(column_names[pair[1]])
        pair_values.append(
            mannwhitneyu(
                fitness_list[pair[0]], fitness_list[pair[1]], alternative="two-sided"
            )[1]
        )
        delta_value = round(
            cliffsDelta(fitness_list[pair[0]], fitness_list[pair[1]])[0], 3
        )
        delta_name = cliffsDelta(fitness_list[pair[0]], fitness_list[pair[1]])[1]
        pair_values.append(str(delta_value) + str(", ") + delta_name)
        for i, p in enumerate(pair_values):
            if not (isinstance(p, str)):
                pair_values[i] = round(pair_values[i], 3)
        rows.append(pair_values)

    log.info("Writing cliff delta data to file: " + plot_name + "_res_p_value.csv")

    with open(plot_name + "_res_p_value_fitness.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)

    rows = [title]
    for pair in combinations(range(0, len(diversity_list)), 2):
        pair_values = []
        pair_values.append(column_names[pair[0]])
        pair_values.append(column_names[pair[1]])
        pair_values.append(
            mannwhitneyu(
                diversity_list[pair[0]],
                diversity_list[pair[1]],
                alternative="two-sided",
            )[1]
        )

        delta_value = round(
            cliffsDelta(diversity_list[pair[0]], diversity_list[pair[1]])[0], 3
        )
        delta_name = cliffsDelta(diversity_list[pair[0]], diversity_list[pair[1]])[1]
        pair_values.append(str(delta_value) + str(", ") + delta_name)
        for i, p in enumerate(pair_values):
            if not (isinstance(p, str)):
                pair_values[i] = round(pair_values[i], 3)
        rows.append(pair_values)

    log.info(f"Writing to {plot_name + '_res_p_value_diversity.csv'}")
    with open(plot_name + "_res_p_value_diversity.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


def plot_convergence(dfs, stats_names, plot_name):
    """
    Function for plotting the convergence of the algorithms
    It takes a list of dataframes and a list of names for the dataframes, and plots the mean and
    standard deviation of the dataframes

    :param dfs: a list of dataframes, each containing the mean and standard deviation of the fitness of
    the population at each generation
    :param stats_names: The names of the algorithms
    """
    fig, ax = plt.subplots()

    plt.xlabel("Number of generations", fontsize=22)
    #plt.xlabel("Number of simulations", fontsize=16)
    #plt.ylabel("Hypervolume", fontsize=22)
    plt.ylabel("Fitness", fontsize=22)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    len_df = np.inf
    for i, df in enumerate(dfs):
        cur_len = len(dfs[i]["mean"])
        if cur_len < len_df:
            len_df = cur_len

    for i, df in enumerate(dfs):
        # x = np.arange(0, len(dfs[i]["mean"]))
        x = np.arange(0, len_df)
        plt.plot(x, dfs[i]["mean"][:len_df], label=stats_names[i])
        plt.fill_between(
            x,
            np.array(dfs[i]["mean"][:len_df] - dfs[i]["std"][:len_df]),
            np.array(dfs[i]["mean"][:len_df] + dfs[i]["std"][:len_df]),
            alpha=0.2,
        )
        plt.legend()

    log.info("Saving plot to " + plot_name + "_convergence.png")
    plt.savefig(plot_name + "_convergence.png", bbox_inches="tight")
    plt.close()


def plot_convergence_time(dfs, stats_names, name):
    """
    Function for plotting the convergence of the algorithms
    It takes a list of dataframes and a list of names for the dataframes, and plots the mean and
    standard deviation of the dataframes
    
    :param dfs: a list of dataframes, each containing the mean and standard deviation of the fitness of
    the population at each generation
    :param stats_names: The names of the algorithms
    """
    print(name)
    fig, ax = plt.subplots() 

    if name == "iter":
        plt.xlabel("Number of iterations", fontsize=22)
    else:
        plt.xlabel("Time, minutes", fontsize=22)

    plt.ylabel("Number of failures", fontsize=22)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid()

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    

    for i, df in enumerate(dfs):
        current_df = dfs[i]
        size_list = []
        for r, run in enumerate(current_df):
            size_list.append(len(current_df[run]))
        num_elements = min(size_list)
 
        mean_values_array = []
        mean_times_array = []
        for r, run in enumerate(current_df):
            values = np.fromiter(current_df[run].values(), dtype=float)[:num_elements]
            time_stamps = np.fromiter(current_df[run].keys(), dtype=float)[:num_elements]#np.array(list(current_df[run].keys()))

            mean_values_array.append((values))
            mean_times_array.append((time_stamps))

        mean_values = np.mean(mean_values_array, axis=0)
        mean_times = np.mean(mean_times_array, axis=0)
        std_values = np.std(mean_values_array, axis=0)

        line_styles = ['-', '--', '-.', ':', 'solid', 'dashed'] 
        plt.plot(mean_times, mean_values, label=stats_names[i], linestyle=line_styles[i % len(line_styles)], linewidth=2)
        #markers = ['o', 's', 'D', 'v', '^', 'p', '*', 'h']
        #plt.scatter(mean_times, mean_values, marker=markers[i % len(markers)], color='black', s=1)
       # plt.scatter(mean_times, mean_values, marker='o', color='black', s=1)
        plt.fill_between(mean_times, np.array(mean_values - std_values), np.array(mean_values + std_values), alpha=0.2)
        plt.legend()
        plt.xlim(0, 120)
        #plt.xticks(np.arange(0, 125, 10))
        

    
    plt.savefig(name + '_convergence.png', bbox_inches='tight')



    plt.close()

def plot_boxplot(data_list, label_list, name, max_range=None, plot_name=""):
    """
     Function for plotting the boxplot of the statistics of the algorithms
    It takes a list of lists, a list of labels, a name, and a max range, and plots a boxplot of the data

    :param data_list: a list of lists, each list containing the data for a particular algorithm
    :param label_list: a list of labels, each label corresponding to the data in the data_list
    :param name: the name of the plot
    :param max_range: the maximum value of the y-axis
    """

    fig, ax1 = plt.subplots()  # figsize=(8, 4)
    #ax1.set_xlabel("Generator", fontsize=20)
    ax1.set_xlabel("Rho value", fontsize=20)
    
    #ax1.set_xlabel("Algorithm", fontsize=22)
    #ax1.set_xlabel("Tool", fontsize=22)


    ax1.set_ylabel(name, fontsize=22)

    ax1.tick_params(axis="both", labelsize=18)

    ax1.yaxis.grid(
        True, linestyle="-", which="both", color="darkgray", linewidth=2, alpha=0.5
    )
    if max_range == None:
        max_vals = [max(x) for x in data_list]
        max_range = max(max_vals) + 0.1*max(max_vals)
        #max_range = max(data_list) + 0.1*max(data_list)

    top = max_range
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.boxplot(data_list, widths=0.6, labels=label_list)

    plt.subplots_adjust(bottom=0.15, left=0.16)

    log.info("Saving box plot: " + plot_name + "_" + name + ".png")

    fig.savefig(plot_name + "_" + name + ".png", bbox_inches="tight")
    plt.close()

def calculate_test_list_novelty(test_list, problem="robot"):
    """
    Calculate the novelty of a test list.

    This function calculates the novelty of a given test list by comparing each pair of tests
    in the list and calculating the novelty score using the `calc_novelty` function.

    Parameters:
    - test_list (list): A list of test objects.
    - problem (str): The problem type. Default is "robot".

    Returns:
    - novelty (float): The average novelty score of the test list.
    """
    novelty_list = []
    for i in combinations(range(0, len(test_list)), 2):
        current1 = test_list[i[0]]  # res.history[gen].pop.get("X")[i[0]]
        current2 = test_list[i[1]]  # res.history[gen].pop.get("X")[i[1]]
        nov = calc_novelty(current1, current2, problem)
        novelty_list.append(nov)
    novelty = sum(novelty_list) / len(novelty_list)
    return novelty

def analyse_all_test(tests_path, stats_names, plot_name):

    stats_paths = []

    print(tests_path)
    for path in tests_path:
        for file in os.listdir(path):
            if "all_tests" in file:
                stats_paths.append(os.path.join(path, file))

    
    fail_num_list_all = []
    tot_tests_num_all = []
    valid_test_num_all = []
    failure_percentage_all = []
    novelty_list_all = []
    dfs = {}
    dfs_time = {}
    
    for i, file in enumerate(stats_paths):
        with open(file, "r", encoding="utf-8") as f:
            all_tests = json.load(f)

        failure_convergence = {}
        fail_num_list = []
        tot_tests_num = []
        valid_test_num = []
        failure_percentage = []
        novelty_list = []   
        for run in all_tests:
            failure_convergence[run] = {}
            fail_num = 0
            tot_tests = 0
            valid_tests = 0
            failed_test_list = []
            n_sim  = all_tests[run][list(all_tests[run].keys())[-1]]["n_sim"]
            #n_sim  = len(list(all_tests[run].keys()))
            #print(n_sim)
            #if "random_rl" in file:
            #    dt = 6500/n_sim
            #else:
            if i==0 or i==2:
                dt = 8000/n_sim
            #if i == 2:
            #    dt = 7800/n_sim
            elif i == 3:
                dt = 7950/n_sim
            else:
                dt = 7750/n_sim
            dt = dt/60 # in minutes
            time_passed = 0
            for tc in all_tests[run]:
                #print(file)
                #print(run)
                #print(tc)
                if all_tests[run][tc]["info_validity"] == "True" : #and all_tests[run][tc]["fitness"] != 0
                    valid_tests += 1
                    time_passed += dt
                    #time_passed = min(120, time_passed)

                    if all_tests[run][tc]["outcome"] == "FAIL":
                        fail_num += 1
                        failed_test_list.append(all_tests[run][tc]["test"])

                    #failure_convergence[run][str(all_tests[run][tc]["n_sim"])] = all_tests[run][tc]["num_failures"]
                    failure_convergence[run][str(time_passed)] = all_tests[run][tc]["num_failures"]
                tot_tests +=1
            fail_num_list.append(fail_num)
            tot_tests_num.append(tot_tests)
            valid_test_num.append(valid_tests/tot_tests)
            failure_percentage.append(fail_num/valid_tests)
            #print(fail_num)
            if "BEAMNG" in file:
                problem = "vehicle"
            else:   
                problem = "robot"
            novelty = calculate_test_list_novelty(failed_test_list, problem=problem)
            novelty_list.append(novelty)
            #print(f"Failure convergence size {len(failure_convergence[run])}")

        fail_num_list_all.append(fail_num_list)
        tot_tests_num_all.append(tot_tests_num)
        valid_test_num_all.append(valid_test_num)
        failure_percentage_all.append(failure_percentage)
        
        novelty_list_all.append(novelty_list)

        dfs[i] = pd.DataFrame(data=failure_convergence)
        dfs[i]["mean"] = dfs[i].mean(axis=1)
        dfs[i]["std"] = dfs[i].std(axis=1)
        #dfs[i].dropna(axis=0, how='all', inplace=True)
        dfs[i] = dfs[i].iloc[:]

        dfs_time[i] = failure_convergence
        #print(f"Size {len(dfs_time[i])}")
        #dfs[i].drop([70:], axis=0)

    #plot_convergence(dfs, stats_names, plot_name)
    plot_convergence_time(dfs_time, stats_names, plot_name)
    print(fail_num_list_all)
    
    
    plot_boxplot(fail_num_list_all, stats_names, "Number of failures", plot_name=plot_name)
    plot_boxplot(tot_tests_num_all, stats_names, "Total tests generated", plot_name=plot_name)
    plot_boxplot(valid_test_num_all, stats_names, "Percentage of valid tests", plot_name=plot_name)
    plot_boxplot(failure_percentage_all, stats_names, "Percentage of failed tests", plot_name=plot_name)
    plot_boxplot(novelty_list_all, stats_names, "Sparseness", plot_name=plot_name)


    build_median_table(fail_num_list_all, novelty_list_all, stats_names, plot_name)
    build_cliff_data(fail_num_list_all, novelty_list_all, stats_names, plot_name)


def analyse(stats_path, stats_names, plot_name):
    """
    Main function for building plots comparing the algorithms
    It takes a list of paths to folders containing the results of the tool runs, and a list of names
    of the runs, and it plots the convergence and the boxplots of the fitness and novelty

    :param stats_path: a list of paths to the folders containing the stats files
    :param stats_names: list of strings, names of the runs
    """
    convergence_paths = []
    stats_paths = []
    hyper_paths = []
    conv_flag = False
    hyper_flag = False
    for path in stats_path:
        for file in os.listdir(path):
            if "conv" in file:
                convergence_paths.append(os.path.join(path, file))
                conv_flag = True
            if "stats" in file:
                stats_paths.append(os.path.join(path, file))
            if "hyper" in file:
                hyper_flag = True
                hyper_paths.append(os.path.join(path, file))


    if conv_flag:
        dfs = {}
        for i, file in enumerate(convergence_paths):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            dfs[i] = pd.DataFrame(data=data)
            dfs[i]["mean"] = dfs[i].mean(axis=1)
            dfs[i]["std"] = dfs[i].std(axis=1)

        plot_convergence(dfs, stats_names, plot_name)


    if hyper_flag:
        dfs = {}
        for i, file in enumerate(hyper_paths):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            dfs[i] = pd.DataFrame(data=data)
            dfs[i]["mean"] = dfs[i].mean(axis=1)
            dfs[i]["std"] = dfs[i].std(axis=1)
        plot_convergence(dfs, stats_names, plot_name + "_hyper")

    fitness_list = []
    best_fitness_list = []
    novelty_list = []
    hyper_list = []
    time_list = []
    max_fitness = 0
    for i, file in enumerate(stats_paths):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results_fitness = []
        results_novelty = []
        results_time = []
        results_best_fitness = []
        results_hyper = []

        for m in range(len(data)):
            fitness_data = [abs(d) for d in data["run" + str(m)]["fitness"]]
            value = max(fitness_data)
            if value > max_fitness:
                max_fitness = value
            best_fitness = value
            results_fitness.extend(fitness_data)  # data["run"+str(m)]["fitness"]
            results_novelty.append(data["run" + str(m)]["novelty"])
            results_best_fitness.append(best_fitness)
            if "hypervolume" in data["run" + str(m)].keys():
                results_hyper.append(data["run" + str(m)]["hypervolume"])

            if "times" in str(data):
                results_time.extend(data["run" + str(m)]["times"])

        fitness_list.append(results_fitness)
        novelty_list.append(results_novelty)
        time_list.append(results_time)
        best_fitness_list.append(results_best_fitness)
        hyper_list.append(results_hyper)

    if results_time:
        max_time = max(max(time_list[0]), max(time_list[1]))
        plot_boxplot(time_list, stats_names, "Time, s", max_time + 0.2, plot_name)
        build_times_table(time_list, stats_names)

    if hyper_list:
        max_hyper = max(max(hyper_list[0]), max(hyper_list[1]))
        plot_boxplot(hyper_list, stats_names, "Hypervolume", max_hyper + 10, plot_name)

    plot_boxplot(
        fitness_list, stats_names, "Fitness", max_fitness + 3, plot_name
    )  # + 2
    plot_boxplot(novelty_list, stats_names, "Diversity", 1.05, plot_name)

    build_median_table(hyper_list, novelty_list, stats_names, plot_name)
    build_cliff_data(hyper_list, novelty_list, stats_names, plot_name)

    compare_mean_best_values_found(hyper_list, stats_names, plot_name)
    compare_p_val_best_values_found(hyper_list, stats_names, plot_name)


def analyse_tools(stats_path, stats_names, plot_name):
    """
    The `analyse_tools` function takes in a list of file paths, a list of statistics names, and a plot
    name, and performs analyse on the data in those files.

    Args:
      stats_path: A list of paths to the directories where the statistics files are located.
      stats_names: stats_names is a list of names for each set of statistics. It is used to label the
    different sets of statistics in the plots and tables.
      plot_name: The name of the plot that will be generated.
    """

    sparseness_list = []
    oob_list = []
    max_sparseness = 0
    max_oob = 0
    for path in stats_path:
        current_sparseness_list = []
        current_oob_list = []
        for root, fd, files in os.walk(path):
            for filename in files:
                #print(filename)
                #print(path)
                #print(root)

                if "oob_stats.csv" in filename:
                    data = pd.read_csv(os.path.join(root, filename))
                    sparseness = float(data["avg_sparseness"])
                    oobs = int(data["total_oob"])
                    current_sparseness_list.append(sparseness)
                    current_oob_list.append(oobs)
                    break
        if max(current_sparseness_list) > max_sparseness:
            max_sparseness = max(current_sparseness_list)
        if max(current_oob_list) > max_oob:
            max_oob = max(current_oob_list)
        sparseness_list.append(current_sparseness_list)
        oob_list.append(current_oob_list)

    plot_boxplot(
        sparseness_list, stats_names, "Sparseness", max_sparseness + 5, plot_name
    )
    plot_boxplot(oob_list, stats_names, "Numbrer of failures", max_oob + 1, plot_name)

    print(f"Sparseness: {sparseness_list}")
    print(f"OOB: {oob_list}")
    build_median_table(oob_list, sparseness_list, stats_names, plot_name)
    build_cliff_data( oob_list, sparseness_list, stats_names, plot_name)


if __name__ == "__main__":
    arguments = parse_arguments()
    setup_logging("log.txt", False)

    stats_path = arguments.stats_path
    stats_names = arguments.stats_names
    plot_name = arguments.plot_name
    tools = arguments.tools

    if tools:
        analyse_tools(stats_path, stats_names, plot_name)
    elif arguments.all_tests:
        analyse_all_test(stats_path, stats_names, plot_name)
    else:
        analyse(stats_path, stats_names, plot_name)

    
