import time
import sys

import pygame.display
from pygame import VIDEORESIZE

from helper_functions import *
from level_setup import *


def run_visualization(surface, scaling=1, FPS=30,
                      obstacles_list_file='object_list_0.csv',
                      drift_ranges_file='drift_ranges_0.csv',
                      wall_list_file='walls_dict.csv',
                      drift_enabled=True, trial=0, attempt=0, n_run=0, code='test'):
    """
    :param surface: argument for specifying pygame.display object
    :param scaling: int (or float) to scale up on-screen visualization
    :param FPS: frames per second. 30 as default value. For smoother animation choose 60.
    :param obstacles_list_file: file for list of obstacles. Has to be in logs repository
    :param drift_ranges_file: file for drift ranges. Has to be in logs repository
    :param wall_list_file: file with wall positions on every y position in level
    :param drift_enabled: drift tiles in level vs. no drift tiles
    :param trial: player movements of which trial (.csv file in logs) to be visualized
    :param attempt: attempts for this specific trial
    :param n_run: total number of runs in this specific experiment / visualization
    :param code: individual code for participant in case of experiment
    """
    # preparing lists of in-game objects from which to draw said objects on screen
    # walls will be the same across all experimental trials
    wall_list = get_walls(wall_list_file)
    wall_list = adjust_walls(wall_list, scaling)

    obstacles_list = get_obstacles(obstacles_list_file)
    obstacles_list = adjust_obstacles(obstacles_list, scaling)

    player_positions_filename = '0_vis.csv'
    player_starting_position, player_positions = get_player_positions(player_positions_filename)
    player_starting_position, player_positions = adjust_player_positions(player_starting_position, player_positions,
                                                                         scaling)

    drift_ranges = get_drift_sections(drift_ranges_file)
    drift_ranges = adjust_drift_sections(drift_ranges, scaling)

    # running through game loop
    level = Level(wall_list=wall_list, obstacles_list=obstacles_list,
                  player_starting_position=player_starting_position, drift_ranges=drift_ranges,
                  drift_enabled=drift_enabled, screen=surface, scaling=scaling, n_run=n_run,
                  trial=trial, attempt=attempt, code=code, FPS=FPS)

    level_done = run_pygame(surface=surface, scaling=scaling, FPS=FPS, player_positions=player_positions, level=level)
    return level_done


def run_pygame(surface, scaling, FPS, player_positions, level):
    """
    :param surface: pygame surface on which objects are drawn
    :param scaling: integer or float which is used to enlarge visualization
    :param FPS: how many frames per second should be drawn
    :param player_positions: only used in simple visualization; where player should be drawn
    :param level: level object which is defined before
    """
    clock = pygame.time.Clock()

    # time onset
    start_time = time.time()
    level_done = False

    quit = False
    while not quit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # FIXME: resizing not working
            if event.type == VIDEORESIZE:
                surface = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        # update time
        time_played = time.time() - start_time
        # print(time_played)

        surface.fill('black')

        # update player position
        current_player_position = player_positions[0]  # not needed but still given in level.run()
        quit, level_done = level.run(time_played, current_player_position, scaling)

        pygame.display.update()
        clock.tick(FPS)

    return level_done
