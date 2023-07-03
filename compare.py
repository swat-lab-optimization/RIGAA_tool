
import os
import argparse
import csv
from itertools import combinations
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import mannwhitneyu

from matplotlib.ticker import MaxNLocator

from rigaa.utils.cliffsDelta import cliffsDelta

def parse_arguments():
    """
    This function parses the arguments passed to the script
    :return: The arguments that are being passed to the program
    """

    print("Parsing the arguments")
    parser = argparse.ArgumentParser(
                    prog = 'compare.py',
                    description = 'A tool for generating test cases for autonomous systems',
                    epilog = "For more information, please visit ")
    # parse a list of arguments
    parser.add_argument('--stats_path', nargs='+', help='The source folders of the metadate to analyze', required=True)
    parser.add_argument('--stats_names', nargs='+', help='The names of the corresponding algorithms', required=True)
    parser.add_argument("--plot_name", help="Name to add to the plots", required=False, default="")
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

    columns  = ["Problem", "Metric"]
    for name in column_names:
        columns.append(name)

    row_0 = [problem, "Mean best value found"]
    for val in best_fitness_list:
        row_0.append(round(np.mean(val), 3))

    rows = [columns, row_0]
    with open(plot_name + "_mean.csv", 'w', newline='') as csvfile:
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
        pair_values.append(mannwhitneyu(best_fitness_list[pair[0]], best_fitness_list[pair[1]], alternative="two-sided")[1])
        delta_value = round(cliffsDelta(best_fitness_list[pair[0]], best_fitness_list[pair[1]])[0], 3)
        delta_name = cliffsDelta(best_fitness_list[pair[0]], best_fitness_list[pair[1]])[1]
        pair_values.append(str(delta_value) + str(", ") + delta_name)
        for i, p in enumerate(pair_values):
            if not(isinstance(p, str)):
               pair_values[i] = round(pair_values[i], 3)
        rows.append(pair_values)

    with open(plot_name + "_res_p_value.csv", 'w', newline='') as csvfile:
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
    columns  = ["Metric"]
    for name in column_names:
        columns.append(name)
    columns.append("p-value")
    columns.append("Effect size")

    row_0 = ["mean generation time, s"]
    for alg in times_list:
        row_0.append(round(np.mean(alg), 3))
    row_0.append(round(mannwhitneyu(times_list[1], times_list[0], alternative="two-sided")[1],3))
    row_0.append(str(round(cliffsDelta(times_list[1], times_list[0])[0],3)) + ", " + str(cliffsDelta(times_list[1], times_list[0])[1]) )
      
    
    row_1 = ["generation time std, s"]
    for alg in times_list:
        row_1.append(np.std(alg))

    rows = [columns, row_0]
    with open("results_time.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)
    


def build_median_table(fitness_list, diversity_list, column_names, plot_name):
    columns  = ["Metric"]
    for name in column_names:
        columns.append(name)

    row_0 = ["Mean fitness"]
    for alg in fitness_list:
        row_0.append(round(np.mean(alg), 3))
      
    row_1 = ["Mean diversity"]
    for alg in diversity_list:
        row_1.append(round(np.mean(alg), 3))

    if (len(fitness_list) == 2) and (len(diversity_list) == 2):
        row_0.append(round(mannwhitneyu(fitness_list[1], fitness_list[0], alternative="two-sided")[1],3))
        row_0.append(round(cliffsDelta(fitness_list[1], fitness_list[0])[0], 3))
        row_0.append(cliffsDelta(fitness_list[1], fitness_list[0])[1])

        row_1.append(round(mannwhitneyu(diversity_list[0], diversity_list[1], alternative="two-sided")[1], 3))
        row_1.append(round(cliffsDelta(diversity_list[0], diversity_list[1])[0], 3))
        row_1.append(cliffsDelta(diversity_list[0], diversity_list[1])[1])
        columns.append("p-value")
        columns.append("Effect size")

    rows = [columns, row_0, row_1]
    with open(plot_name + "_res.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)

def build_cliff_data(fitness_list, diversity_list, column_names, plot_name):
    title = ["A", "B", "p-value", "Effect size"]
    rows = [title]
    for pair in combinations(range(0, len(fitness_list)), 2):
        pair_values = []
        pair_values.append(column_names[pair[0]])
        pair_values.append(column_names[pair[1]])
        pair_values.append(mannwhitneyu(fitness_list[pair[0]], fitness_list[pair[1]], alternative="two-sided")[1])
        delta_value = round(cliffsDelta(fitness_list[pair[0]], fitness_list[pair[1]])[0], 3)
        delta_name = cliffsDelta(fitness_list[pair[0]], fitness_list[pair[1]])[1]
        pair_values.append(str(delta_value) + str(", ") + delta_name)
        for i, p in enumerate(pair_values):
            if not(isinstance(p, str)):
               pair_values[i] = round(pair_values[i], 3)
        rows.append(pair_values)

    with open(plot_name + "_res_p_value_fitness.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)

    rows = [title]
    for pair in combinations(range(0, len(diversity_list)), 2):
        pair_values = []
        pair_values.append(column_names[pair[0]])
        pair_values.append(column_names[pair[1]])
        pair_values.append(mannwhitneyu(diversity_list[pair[0]], diversity_list[pair[1]], alternative="two-sided")[1])
        #pair_values.append(cliffsDelta(diversity_list[pair[0]], diversity_list[pair[1]])[0])
        #pair_values.append(cliffsDelta(diversity_list[pair[0]], diversity_list[pair[1]])[1])
        delta_value = round(cliffsDelta(diversity_list[pair[0]], diversity_list[pair[1]])[0], 3)
        delta_name = cliffsDelta(diversity_list[pair[0]], diversity_list[pair[1]])[1]
        pair_values.append(str(delta_value) + str(", ") + delta_name)
        for i, p in enumerate(pair_values):
            if not(isinstance(p, str)):
               pair_values[i] = round(pair_values[i], 3)
        rows.append(pair_values)
    
    with open(plot_name + "_res_p_value_diversity.csv", 'w', newline='') as csvfile:
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

    plt.xlabel("Number of generations", fontsize=16)
    plt.ylabel("Fitness", fontsize=16)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.grid()
    
    len_df = np.inf
    for i, df in enumerate(dfs):
        cur_len = len(dfs[i]["mean"])
        if cur_len < len_df:
            len_df = cur_len

    for i, df in enumerate(dfs):
        #x = np.arange(0, len(dfs[i]["mean"]))
        x = np.arange(0, len_df)
        plt.plot(x, dfs[i]["mean"][:len_df], label=stats_names[i])
        plt.fill_between(x, np.array(dfs[i]["mean"][:len_df] - dfs[i]["std"][:len_df]), np.array(dfs[i]["mean"][:len_df] + dfs[i]["std"][:len_df]), alpha=0.2)
        plt.legend()
    plt.savefig(plot_name + '_convergence.png', bbox_inches='tight')
    plt.close()

def plot_boxplot(data_list, label_list, name, max_range, plot_name):
    """
     Function for plotting the boxplot of the statistics of the algorithms
    It takes a list of lists, a list of labels, a name, and a max range, and plots a boxplot of the data
    
    :param data_list: a list of lists, each list containing the data for a particular algorithm
    :param label_list: a list of labels, each label corresponding to the data in the data_list
    :param name: the name of the plot
    :param max_range: the maximum value of the y-axis
    """

    fig, ax1 = plt.subplots() #figsize=(8, 4)
    ax1.set_xlabel('Algorithm', fontsize=18)
    #ax1.set_xlabel('Generator', fontsize=20)
    #ax1.set_xlabel('Epsilon value', fontsize=20)

    ax1.set_ylabel(name, fontsize=18)
    

    ax1.tick_params(axis="both", labelsize=14)
    
    ax1.yaxis.grid(True, linestyle='-', which='both', color='darkgray', linewidth=2, alpha=0.5)

    top = max_range
    bottom = 0
    ax1.set_ylim(bottom, top)
    ax1.boxplot(data_list, widths=0.55, labels=label_list)

    plt.subplots_adjust(bottom=0.15, left=0.16)

    fig.savefig(plot_name + "_" + name + ".png", bbox_inches='tight')
    plt.close()


def main(stats_path, stats_names, plot_name):
    """
    Main function for building plots comparing the algorithms
    It takes a list of paths to folders containing the results of the tool runs, and a list of names
    of the runs, and it plots the convergence and the boxplots of the fitness and novelty
    
    :param stats_path: a list of paths to the folders containing the stats files
    :param stats_names: list of strings, names of the runs
    """
    convergence_paths = []
    stats_paths = []
    conv_flag = False
    for path in stats_path:
        for file in os.listdir(path):
            if "conv" in file:
                convergence_paths.append(os.path.join(path, file))
                conv_flag = True
            if "stats" in file:
                stats_paths.append(os.path.join(path, file))

    if conv_flag:    
        dfs = {}
        for i, file in enumerate(convergence_paths):
            with open(file, 'r', encoding="utf-8") as f:
                data = json.load(f)
            dfs[i] = pd.DataFrame(data=data)
            dfs[i]["mean"] = dfs[i].mean(axis=1)
            dfs[i]["std"] = dfs[i].std(axis=1)
        
        plot_convergence(dfs, stats_names, plot_name)

    fitness_list = []
    best_fitness_list = []
    novelty_list = []
    time_list  = []
    max_fitness = 0
    for i, file in enumerate(stats_paths):
        with open(file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        results_fitness = []
        results_novelty = []
        results_time = []
        results_best_fitness = []
        
        for m in range(len(data)):
            fitness_data = [abs(d) for d in data["run"+str(m)]["fitness"]]
            value = max(fitness_data)
            if value > max_fitness:
                max_fitness = value
            best_fitness = value
            results_fitness.extend(fitness_data)  #data["run"+str(m)]["fitness"]
            results_novelty.append(data["run"+str(m)]["novelty"])
            results_best_fitness.append(best_fitness)
            
            if "times" in str(data):
                results_time.extend(data["run"+str(m)]["times"])

        fitness_list.append(results_fitness)
        novelty_list.append(results_novelty)
        time_list.append(results_time) 
        best_fitness_list.append(results_best_fitness)
    
    #max_time = max(max(time_list[0]), max(time_list[1]))
    
    #plot_boxplot(fitness_list, stats_names, "Fitness", max_fitness+ 2, plot_name)
    #plot_boxplot(novelty_list, stats_names, "Diversity", 1.05, plot_name)
    
    #plot_boxplot(time_list, stats_names, "Time, s", max_time+0.2)
    #print(results_time)

    #build_times_table(time_list, stats_names)

    #build_median_table(fitness_list, novelty_list, stats_names, plot_name)
    #build_cliff_data(fitness_list, novelty_list, stats_names, plot_name)
    compare_mean_best_values_found(best_fitness_list, stats_names, plot_name)
    compare_p_val_best_values_found(best_fitness_list, stats_names, plot_name)



if __name__ == "__main__":
    arguments = parse_arguments()
    stats_path = arguments.stats_path
    stats_names = arguments.stats_names
    plot_name = arguments.plot_name

    main(stats_path, stats_names, plot_name)