import os
import ast
import csv
from config import edge, agent_size_x, agent_size_y


def get_obstacles_lists(filename: str, which_list='first'):
    """
    :param filename: ...
    :param which_list: 'all' vs 'first' vs 'last' vs int
    function read the obstacles lists csv. The function takes two arguments: filename and which_list,
    whereas the latter is optional. As filename the actual name of the file in the logs directory should be passed.
    The function will get the path itself. The file must be in the logs directory. Otherwise it won't find it.
    The which_list argument can be passed as string with all vs. first vs. last or an integer as string or integer.
    This will determine the list(s) that are actually returned. Lastly this function will return a bool. It is true for
    when multiple lists are returned and false if a single list is returned.
    """
    obstacles_lists = []
    complete_path = os.getcwd() + '/logs/' + filename
    with open(complete_path, mode="r") as file:
        for i, line in enumerate(file):
            cut = len(str(i)) + 2
            content_list = line[cut:-2]
            list_of_obstacle_dicts = ast.literal_eval(content_list)
            current_obstacles_list = list(list_of_obstacle_dicts)
            obstacles_lists.append(current_obstacles_list)
    if which_list == 'all':
        return obstacles_lists, True
    elif which_list == 'first':
        return obstacles_lists[0], False
    elif which_list == 'last':
        return obstacles_lists[-1], False
    else:
        j = int(which_list)
        return obstacles_lists[j], False


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
                player_starting_position = [int(float(line[0])), int(float(line[1]))-1]
            if not counter == 0:
                player_positions.append([int(float(line[0])), int(float(line[1]))])
            counter += 1
    return player_starting_position, player_positions


def get_wall_positions(filename: str):
    """
    :param filename: filename of .txt file containing positions for all wall tiles. file must be in logs directory
    :return wall_dict: a dict easily accessible to grab individual positions of wall tiles
    """
    complete_path = os.getcwd() + '/logs/' + filename
    with open(complete_path, mode="r") as file:
        for i, line in enumerate(file):
            # indexing due to preceding and ending '"' which prohibits literal_eval to generate dict
            wall_dict = ast.literal_eval(line[1:-2])
    return wall_dict


def get_drift_ranges(filename: str, level=0):
    """
    :param filename: .csv file must be in /logs
    :param level: in case of multiple level drift ranges, indicate level of which the drift ranges to grab
    """
    drift_ranges_dict = []
    complete_path = os.getcwd() + '/logs/' + filename
    with open(complete_path, mode="r") as file:
        for i, line in enumerate(file):
            cut = len(str(i)) + 2
            content_list = line[cut:-2]
            list_of_drift_ranges_dict = ast.literal_eval(content_list)
            current_drift_ranges_list = list(list_of_drift_ranges_dict)
            drift_ranges_dict.append(current_drift_ranges_list)
    return drift_ranges_dict[level]


def update_environment(obstacles_list, scaling):
    """
    :param obstacles_list: all obstacles (x,y) which are in environment
    :param scaling: int to scale up on-screen visualization
    :return obstacles_list: updated obstacles_list y of all obstacles reduced by 1*scaling
    """
    for i in obstacles_list:
        i['y'] = i['y'] - (1 * scaling)
    return obstacles_list


def adjust_wall_list(wall_list, scaling):
    """
    :param wall_list:
    :param scaling: int to scale up on-screen visualization
    :return updated wall_list
    """
    for i in range(1, len(wall_list)+1):
        wall_list[str(i)][0] = (wall_list[str(i)][0] + edge) * scaling
        wall_list[str(i)][1] = ((wall_list[str(i)][1] - 1) + edge) * scaling
    return wall_list


def adjust_obstacles_list(obstacles_list, scaling):
    """
    :param
    :return
    """
    for i in obstacles_list:
        i['x'] = ((i['x'] - 1) + edge) * scaling
        i['y'] = (i['y'] - 1) * scaling
        i['size'] = i['size'] * scaling
    return obstacles_list


def adjust_player_positions(player_starting_position, player_positions, scaling, tiny_visualization=False):
    """
    param
    return
    """
    adjusted_player_starting_position = [(player_starting_position[0] + edge - 0.5*agent_size_x)*scaling,
                                         player_starting_position[1]*scaling]
    for i in player_positions:
        i[0] = ((i[0] - agent_size_x) + edge) * scaling
        if tiny_visualization:
            i[1] = ((i[1] - agent_size_x) + edge) * scaling
        else:
            i[1] = adjusted_player_starting_position[1] - agent_size_y
    return adjusted_player_starting_position, player_positions


def adjust_drift_ranges(drift_ranges_list, scaling):
    """
    param
    return
    """
    for i in drift_ranges_list:
        i[0] = (i[0] - 1) * scaling
        i[1] = (i[1] - 1) * scaling
    return drift_ranges_list
