import os
import ast
import csv
import numpy as np
import pandas as pd
from config import edge, agent_size_x, agent_size_y


def get_player_positions(filename: str):
    """
    :param filename: file must be in logs directory
    :return: player_starting_positions, all positions of the agent within trial
    """
    complete_path = os.getcwd() + '/logs/' + filename
    with open(complete_path, mode="r") as file:
        csv_file = csv.reader(file)
        player_starting_position = []
        player_positions = []
        counter = 0
        for line in csv_file:
            if counter == 1:
                # convert from str to float to int
                player_starting_position = [int(float(line[0])), int(float(line[1])) - 1]
            if not counter == 0:
                player_positions.append([int(float(line[0])), int(float(line[1]))])
            counter += 1
    return player_starting_position, player_positions


def update_environment(obstacles_list, scaling):
    """
    :param obstacles_list: all obstacles (x,y) which are in environment
    :param scaling: int to scale up on-screen visualization
    :return obstacles_list: updated obstacles_list y of all obstacles reduced by 1*scaling
    """
    for i in obstacles_list:
        i['y'] = i['y'] - (1 * scaling)
    return obstacles_list


def adjust_player_positions(player_starting_position, player_positions, scaling, tiny_visualization=False):
    """
    param
    return
    """
    adjusted_player_starting_position = [(player_starting_position[0] + edge - 0.5 * agent_size_x) * scaling,
                                         player_starting_position[1] * scaling]
    for i in player_positions:
        i[0] = ((i[0] - agent_size_x) + edge) * scaling
        if tiny_visualization:
            i[1] = ((i[1] - agent_size_x) + edge) * scaling
        else:
            i[1] = adjusted_player_starting_position[1] - agent_size_y
    return adjusted_player_starting_position, player_positions


# new functions
def get_walls(filename: str):
    """
    :param filename: filename of .txt file containing positions for all wall tiles. file must be in logs directory
    :return simple_dict: a dict easily accessible to grab individual positions of wall tiles
    """
    complete_path = os.getcwd() + '/logs/' + filename

    walls_df = pd.read_csv(complete_path, usecols=[1], header=None)
    walls_dict = walls_df.to_dict(orient='index')

    simple_dict = {key + 1: ast.literal_eval(value[1]) for key, value in walls_dict.items()}  # +1 for starting at 1

    return simple_dict


def adjust_walls(walls_dict, scaling):
    """
    :param walls_dict:
    :param scaling: int to scale up on-screen visualization
    :return updated walls_dict
    """
    for i in range(1, len(walls_dict) + 1):
        walls_dict[i][0] = (walls_dict[i][0] + edge) * scaling
        walls_dict[i][1] = ((walls_dict[i][1] - 1) + edge) * scaling

    return walls_dict


def get_obstacles(filename: str, colnames=['x', 'y', 'size']):
    complete_path = os.getcwd() + '/logs/' + filename

    obstacles_df = pd.read_csv(complete_path, names=colnames, header=None)
    obstacles_dict = obstacles_df.to_dict(orient='index')

    return obstacles_dict


def adjust_obstacles(obstacles_dict, scaling):
    """
    :param
    :return
    """
    for i in obstacles_dict:
        obstacles_dict[i]['x'] = ((obstacles_dict[i]['x'] - 1) + edge) * scaling
        obstacles_dict[i]['y'] = (obstacles_dict[i]['y'] - 1) * scaling
        obstacles_dict[i]['size'] = obstacles_dict[i]['size'] * scaling
    return obstacles_dict


def get_drift_sections(filename: str, colnames=['y_start', 'y_end', 'direction', 'visibility', 'fake']):
    complete_path = os.getcwd() + '/logs/' + filename
    drift_df = pd.read_csv(complete_path, names=colnames, header=None)

    drift_dict = drift_df.to_dict(orient='index')

    return drift_dict


def adjust_drift_sections(drift_dict, scaling):
    """
    param
    return
    """
    for i in drift_dict:
        drift_dict[i]['y_start'] = (drift_dict[i]['y_start'] - 1) * scaling
        drift_dict[i]['y_end'] = (drift_dict[i]['y_end'] - 1) * scaling
    return drift_dict


# Functions for converting distances in pixel on screen to visual degrees and back
def pixel_to_degree(
    distance_on_screen_pixel, mm_per_pixel=595 / 1920, distance_to_screen_mm=700
):
    """
    calculate the visual degrees of a distance (saccade amplitude, on screen distance between objects)
    setup:
     - screen_width_in_mm=595
     - screen_height_in_mm=335
     - pixels_screen_width=1920
     - pixels_screen_height=1080
     - distance_to_screen_in_mm=700
    """
    distance_on_screen_mm = float(distance_on_screen_pixel) * mm_per_pixel

    visual_angle_in_radians = np.arctan(distance_on_screen_mm / distance_to_screen_mm)

    return np.rad2deg(visual_angle_in_radians)


def degree_to_pixel(target_size=1, mm_per_pixel=595 / 1920, distance_to_screen_mm=700):
    """
    target_size in visual degrees

    calculate the amount of pixel on screen to cover target size in visual degrees (stimulus size)
    setup:
     - screen_width_in_mm=595
     - screen_height_in_mm=335
     - pixels_screen_width=1920
     - pixels_screen_height=1080
     - distance_to_screen_in_mm=700
    """
    target_size_in_radians = np.deg2rad(target_size)

    distance_on_screen_pixel = (
        np.tan(target_size_in_radians) * distance_to_screen_mm / mm_per_pixel
    )

    return distance_on_screen_pixel
