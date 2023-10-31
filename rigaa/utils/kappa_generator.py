import math

from shapely import geometry, affinity

import random
import numpy as np
import matplotlib.pyplot as plt 
from shapely.geometry import LineString
from descartes import PolygonPatch
import time

#from road_validity_check import is_valid_road
from rigaa.utils.road_validity_check import is_valid_road

class KappaRoadGenerator():
    '''
    Class to generate a road based on a kappa function.
    '''
    def __init__(self, map_size, theta0, segment_length, margin):

        self.number_of_points = 20
        self.global_bound = 0.06232
        self.local_bound = 0.05
        self.map_size = map_size
        self.theta0 = theta0
        self.segment_length = segment_length
        self.margin = margin
        self.map_offset = 5

    def get_next_kappa(self, last_kappa):
        """
        Generates a new kappa value based on the previous value.

        Args:
            previous_kappa: the previous kappa value

        Returns:
            a new kappa value
        """
        return random.choice(np.linspace(max(-self.global_bound, last_kappa - self.local_bound),
                                         min(self.global_bound, last_kappa + self.local_bound)))

    def generate_random_test(self):
        """ 
        Generates a test using frenet framework to determine the curvature of the points.

        Returns:
            a list of kappa values and its cartesian representation.
        """
        points = self.number_of_points + random.randint(-5, 5)
        # Producing randomly generated kappas for the given setting.
        kappas = [0.0] * points
        for i in range(len(kappas)):
            kappas[i] = self.get_next_kappa(kappas[i - 1])

        self.kappas = kappas

        return kappas

    def frenet_to_cartesian(self, x0, y0, theta0, ss, kappas):
        """Trapezoidal integration to compute Cartesian coordinates from given curvature values."""
        xs = np.zeros(len(kappas))
        ys = np.zeros(len(kappas))
        thetas = np.zeros(len(kappas))
        xs[0] = x0
        ys[0] = y0
        thetas[0] = theta0
        for i in range(thetas.shape[0] - 1):
            ss_diff_half = (ss[i + 1] - ss[i]) / 2.0
            thetas[i + 1] = thetas[i] + (kappas[i + 1] + kappas[i]) * ss_diff_half
            xs[i + 1] = xs[i] + (np.cos(thetas[i + 1]) + np.cos(thetas[i])) * ss_diff_half
            ys[i + 1] = ys[i] + (np.sin(thetas[i + 1]) + np.sin(thetas[i])) * ss_diff_half
        return (xs, ys)


    def kappas_to_road_points(self, kappas) -> np.array:
        """
        Args:
            kappas: list of kappa values
            frenet_step: The distance between to points.
            theta0: The initial angle of the line. (1.57 == 90 degrees)
        Returns:
            road points in cartesian coordinates
        """
        # Using the bottom center of the map.
        y0 = self.map_size / 2#self.margin
        x0 = self.map_size / 2
        theta0 = self.theta0
        ss = np.cumsum([self.segment_length] * len(kappas)) - self.segment_length
        # Transforming the frenet points to cartesian
        (xs, ys) = self.frenet_to_cartesian(x0, y0, theta0, ss, kappas)

        return np.column_stack([xs, ys])

    def visualize_road(self, road_points, save_path : str ="test.png"):

        """
        It takes a list of states, and plots the road and the car path

        Args:
          states: a list of tuples, each tuple is a state of the car.
          save_path: The path to save the image to. Defaults to test.png
        """


        road_points = list(road_points)

        intp_points = road_points#interpolate_road(road_points)

        fig, ax = plt.subplots(figsize=(8, 8))
        road_x = []
        road_y = []

        for p in intp_points:
            road_x.append(p[0])
            road_y.append(p[1])

        

        top = self.map_size
        bottom = 0

        road_line = LineString(road_points)
        ax.plot(road_x, road_y, "yo--", label="Road")
        # Plot the road as a line with custom styling
        #ax.plot(*road_line.xy, color='gray', linewidth=10.0, solid_capstyle='round', zorder=4)
        
        
        road_poly = LineString([(t[0], t[1]) for t in intp_points]).buffer(
            4.0, cap_style=2, join_style=2
        )
        road_patch = PolygonPatch(
            (road_poly), fc="gray", ec="dimgray"
        )  # ec='#555555', alpha=0.5, zorder=4)
        ax.add_patch(road_patch)
        

        # Set axis limits to show the entire road
        ax.set_xlim(road_line.bounds[0], road_line.bounds[2])
        ax.set_ylim(road_line.bounds[1], road_line.bounds[3])

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=16)
        ax.set_ylim(bottom, top)
        plt.ioff()
        ax.set_xlim(bottom, top)
        ax.legend()
        fig.savefig(save_path)
        plt.close(fig)


    def generate_road(self):
        """
        Generates a road using the kappa function.

        Returns:
            a list of road points.
        """
        kappas = self.generate_random_test()
        road_points = self.kappas_to_road_points(kappas)
        while not(is_valid_road(road_points, self.map_size, self.map_offset)):
            kappas = kappas[:-2]
            road_points = self.kappas_to_road_points(kappas)
        
        return road_points

if __name__ == "__main__":
    gen =  KappaRoadGenerator(200, 1.57, 10, 8)
    start = time.time()
    road = gen.generate_road()
    print("Gen_time", time.time() - start)
    gen.visualize_road(road)
    print(road)
    print(gen.kappas)
