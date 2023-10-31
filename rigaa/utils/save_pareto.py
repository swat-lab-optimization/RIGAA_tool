
import os
import logging as log
import config as cf
from datetime import datetime
import matplotlib.pyplot as plt

def save_pareto_front_img(pareto, rest, problem, run, algo, name):
    """
    It takes a pareto optimal solutions, a problem, and a run number, and then it saves the images of the test suite
    in the images folder

    Args:
      test_suite: a list of pareto optimal parameters
      problem: the problem to be solved. Can be "robot", "vehicle", cart_pole" or "lunar_lander"
      run: the number of the runs
      algo: the algorithm used to generate the test suite. Can be "random", "ga", "nsga2",
    """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y")

    images_path = dt_string + "_" + cf.files["images_path"] +  "_pareto_" + algo + "_" + problem + "_" + name

    if not os.path.exists(images_path):
        os.makedirs(images_path)
    #if not os.path.exists(os.path.join(images_path, "run" + str(run))):
    #    os.makedirs(os.path.join(images_path, "run" + str(run))) 
    path = os.path.join(images_path, "run" + str(run) + ".png")

    fig = plt.figure(figsize=(8, 6))
    # Get x and y coordinates from the data array
    f1 = pareto[:, 0]
    f2 = pareto[:, 1]

    f1_rest = rest[:, 0]
    f2_rest = rest[:, 1]

    # Create a scatter plot
    plt.scatter(f1, f2, label='Pareto Front')
    plt.scatter(f1_rest, f2_rest, label='Rest of the Data')

    # Add a legend
    plt.legend()
    # Add labels and title to the plot
    plt.xlabel('f1, fitness', fontsize=18)
    plt.ylabel('f2, diversity', fontsize=18)
    plt.title('Pareto optimal solutions', fontsize=20)
    plt.close(fig)

    # Show the plot
    
    fig.savefig(path, dpi=300, bbox_inches='tight')
    
    log.info(
        "Pareto fronts saved in %s", images_path
    )

