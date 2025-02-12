import pygame
import numpy as np
import pandas as pd
import random
from ingame_objects.comets import Comet
from ingame_objects.player import Player
from ingame_objects.walls import Wall
from ingame_objects.drift_tiles import DriftTile
from ingame_objects.particles import Particle
from ingame_objects.lines import Line
from displays import display_soc_question, display_prior_question
from helper_functions import degree_to_pixel
from draw_transparent_shapes import draw_rect_alpha, draw_polygon_alpha, draw_circle_alpha
from config import *
# agent
from action_planner.action_planner import ActionPlanner
from action_planner.CCL_action_selection import select_action_goal, sample_gaze_location, select_drift_path
from action_planner.helper_functions import load_parameters_dict

# bad practice but here we go
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


display_keys = False
question_soc = True


class Level:
    def __init__(self, wall_list, obstacles_list, player_starting_position, drift_ranges, screen, scaling, code, FPS=30,
                 n_run=0, trial=0, attempt=0, drift_enabled=False):

        # experiment information
        self.code = code

        # prior belief for level
        self.prior = None
        # sense of control during level
        self.SoC = None

        # level_setup
        self.trial = trial
        self.attempt = attempt
        self.display_surface = screen
        self.level_size_y = 'undetermined'
        self.setup_level(wall_list, obstacles_list, player_starting_position, drift_ranges, drift_enabled, scaling)
        # environment movement around agent
        self.direction = pygame.math.Vector2(0, 0)
        # environmentally imposed drift
        self.drift = pygame.math.Vector2(0, 0)
        # player input
        self.current_input = None  # None vs. 'Right' vs. 'Left'
        # total horizontal movement combined of agent imposed direction and environmentally imposed drift
        self.horizontal_movement = 0
        # transparency for keys being pressed
        self.transparency_left = 90
        self.transparency_right = 90

        # listing visible obstacles in every instance, as well as adjacent wall tiles and last wall tile positions
        self.visible_obstacles = []
        self.visible_drift_tiles = []
        self.adjacent_wall_tiles_x_pos = []
        self.last_wall_pos = []

        # Whether drift tiles appear and actually impose drift depends on this variable
        self.drift_enabled = drift_enabled

        # n_run indicating the number of trials after starting the program (used to differentiate data files)
        self.n_run = n_run

        # Collision threshold; number of frames with player colliding to stop game
        self.frames_collision_threshold = FPS / 10
        # frames with player colliding
        self.frames_with_collision = 0

        self.time_played = 0
        self.FPS = FPS
        self.level_done = False
        self.quit = False
        # threshold for replaying trial - if time_played above threshold => no replay
        self.replay_threshold = 1000  # 25; in s
        if 'training' in str(self.trial):  # TRY CONTAIN
            self.replay_threshold = 1000  # arbitrarily high threshold that is never reached

        # Initiate agent from class ActionPlanner for simulating dynamic decision-making
        parameters = load_parameters_dict("action_planner/Data/parameters.txt")
        self.agent = ActionPlanner(free_parameters=parameters,
                                   initial_position_x=self.player.sprite.rect.x + scaling,
                                   observation_space_in_pixel=[observation_space_size_x*scaling, observation_space_size_y*scaling])
        self.convolutionGranularity = int(self.agent.parameters['convolutionGranularity'])

        self.action_goal_selected = False

        # pandas Dataframe in which data of each frame will be stored
        self.columns = ['trial', 'attempt', 'time_played', 'level_size_y', 'player_pos', 'collision', 'current_input',
                        'drift_enabled', 'current_drift', 'level_done',
                        'visible_obstacles', 'last_walls_tile', 'adjacent_wall_tiles_x_pos', 'visible_drift_tiles',
                        'prior', 'SoC',
                        'action_goal_selected', 'action_goal_x', 'action_goal_y']

        self.data = pd.DataFrame(columns=self.columns)

        # convert to boolean where necessary
        self.data["collision"] = self.data["collision"].astype(bool)
        self.data["drift_enabled"] = self.data["drift_enabled"].astype(bool)
        self.data["level_done"] = self.data["level_done"].astype(bool)

    def setup_level(self, wall_list, obstacles_list, player_starting_position, drift_ranges, drift_enabled, scaling):
        self.walls = pygame.sprite.Group()
        self.comets = pygame.sprite.Group()
        self.drift_tiles = pygame.sprite.Group()
        self.player = pygame.sprite.GroupSingle()
        self.particles = pygame.sprite.Group()
        self.bottom_edge = pygame.sprite.GroupSingle()
        self.finish_line = pygame.sprite.GroupSingle()

        for i in range(1, len(wall_list) + 1):
            # left wall
            left_wall_x_pos = (wall_list[i][0])
            left_wall = Wall((left_wall_x_pos, i * scaling), wall_size, scaling)
            # i*scaling will result in the correct y-coord of the wall

            # right wall
            right_wall_x_pos = (wall_list[i][1])
            right_wall = Wall((right_wall_x_pos, i * scaling), wall_size, scaling)

            # add both walls to sprite group
            self.walls.add(left_wall, right_wall)

        # determine level_size_y based on walls
        last_wall_tile = self.walls.sprites()[-1]
        self.level_size_y = last_wall_tile.rect.y + wall_size * scaling

        for key in obstacles_list:  # arguments in Comet(): x-pos, y-pos, tile_size
            comet_sprite = Comet(obstacles_list[key]['x'], obstacles_list[key]['y'], obstacles_list[key]['size'])
            self.comets.add(comet_sprite)

        player_appearance = [player_starting_position[0], (player_starting_position[1] - pre_trial_steps * scaling)]
        player_sprite = Player(player_appearance, agent_size_x, agent_size_y, scaling)
        self.player.add(player_sprite)

        if drift_enabled:
            for i in range(len(drift_ranges)):
                # drift_info[0]: y_start, [1]: y_end, [2]: direction+magnitude, [3]: visibility
                drift_tile = DriftTile(drift_ranges[i]['y_start'], drift_ranges[i]['y_end'], drift_tile_size_x, observation_space_size_x, edge,
                                       drift_ranges[i]['direction'], drift_ranges[i]['visibility'], scaling)
                self.drift_tiles.add(drift_tile)

        for _ in range(int(last_wall_tile.rect.y / scaling * 2.5)):
            x_pos = np.random.uniform(low=edge * scaling, high=level_size_x * scaling + edge * scaling, size=1)
            y_pos = np.random.uniform(low=0, high=self.level_size_y, size=1)
            particle_tile = Particle((x_pos[0], y_pos[0]), random.choice(particle_sizes), scaling)
            self.particles.add(particle_tile)

        # grey edge at bottom of screen limiting observation window (do NOT update in .run)
        bottom_edge_tile = Line([0, (observation_space_size_y - bottom_edge) * scaling],
                                [(level_size_x + 2 * edge) * scaling, bottom_edge * scaling])
        self.bottom_edge.add(bottom_edge_tile)

        finish_line_tile = Line(pos=[edge * scaling, last_wall_tile.rect.y],
                                size=[level_size_x * scaling, scaling], col="seagreen")
        self.finish_line.add(finish_line_tile)

    def get_input(self):

        # reset transparency for keys
        self.transparency_left = 90
        self.transparency_right = 90

        if self.agent.action == 'Right':
            self.direction.x = -1
            self.transparency_right = 150
        elif self.agent.action == 'Left':
            self.direction.x = 1
            self.transparency_left = 150
        else:  # self.agent.action is None
            self.direction.x = 0

    def update(self):
        # create observation space of agent from subsection of display between boarders
        player = self.player.sprite

        # get reference point
        reference_point = (self.walls.sprites()[-2].rect.x + scaling, 236)
        # 236 being player.sprite.rect.bottom after approach at the start of level
        # 208 is player.sprite.rect.top; player included in goal-driven visual environment subsection

        if len(self.visible_drift_tiles) > 0:
            for drift_tile in self.visible_drift_tiles:
                if (drift_tile[1] > player.rect.bottom) & \
                        (drift_tile[1]+(15*scaling) < (observation_space_size_y-bottom_edge)*scaling):
                    # the first drift tile that is below agent
                    if drift_tile[0] < player.rect.x:
                        self.agent.drift_direction = 1
                    else:
                        self.agent.drift_direction = -1
                    drift_situation = reference_point[0], drift_tile[1], 532, 15*scaling  # 15=y size of drift
                    drift_situation_surface = self.display_surface.subsurface(drift_situation)
                    drift_surface_array = np.transpose(pygame.surfarray.array_green(drift_situation_surface))
                    drift_surface_array[drift_surface_array > 1] = 1

                    select_drift_path(PAR=self.agent.parameters,
                                      observation_in_pixel=drift_surface_array,
                                      drift_prior=self.agent.drift_prior,
                                      drift_direction=self.agent.drift_direction,
                                      min_percentage_for_rejection=self.agent.min_percentage_for_rejection)

                    break  # only first drift situation is planned

        #
        observation_space = reference_point[0], reference_point[1], 532, 394
        surface_subsection = self.display_surface.subsurface(observation_space)
        surface_array = np.transpose(pygame.surfarray.array_green(surface_subsection))
        surface_array[surface_array > 1] = 1

        # gaze location
        if self.agent.action_goal:
            self.agent.gaze_location = sample_gaze_location(
                HL_SoC=self.agent.HL_SoC,
                observation_in_pixel=surface_array,
                action_goal_x=self.agent.action_goal[0],
                action_goal_y=self.agent.action_goal[1],
                reference=reference_point)
        else:
            self.agent.gaze_location = sample_gaze_location(
                HL_SoC=self.agent.HL_SoC,
                observation_in_pixel=surface_array,
                action_goal_x=266,
                action_goal_y=50,
                reference=reference_point)

        # update agents predictions if horizontal movement present (due to action or drift)
        if self.horizontal_movement != 0:
            self.agent.perceived_step_size = abs(self.horizontal_movement)
            self.agent.prediction_error()

        # generate action_goal if none is applied OR assess action goal (vertically and horizontally) if one is applied
        self.action_goal_selected = False
        if self.agent.action_goal is None:
            self.agent.action_goal, self.agent.action_goal_col, _, self.agent.HL_SoC = \
                select_action_goal(PAR=self.agent.parameters,
                                   HL_SoC=self.agent.HL_SoC,
                                   observation_in_pixel=surface_array,
                                   reference=reference_point,
                                   agent_pos_x=self.agent.agent_pos_x,
                                   min_percentage_for_rejection=self.agent.min_percentage_for_rejection)
            self.action_goal_selected = True
        else:
            # monitoring application of selected action goal
            self.agent.assess_action_goal(observation_in_pixel=surface_array, reference=reference_point, radius=scaling)
            if self.agent.action_goal is None:
                self.agent.action_goal, self.agent.action_goal_col, _, self.agent.HL_SoC = \
                    select_action_goal(PAR=self.agent.parameters,
                                       HL_SoC=self.agent.HL_SoC,
                                       observation_in_pixel=surface_array,
                                       reference=reference_point,
                                       agent_pos_x=self.agent.agent_pos_x,
                                       min_percentage_for_rejection=self.agent.min_percentage_for_rejection)
                self.action_goal_selected = True

        # get input from agent
        self.get_input()

        # apply horizontal movement of agent in environment (environment moves around agent)
        self.horizontal_movement = self.direction.x + self.drift.x  # compute horizontal movement with drift

    def check_for_collision(self):
        player = self.player.sprite
        self.last_wall_pos = [self.walls.sprites()[-2].rect.x, self.walls.sprites()[-1].rect.x]

        # checking for general collision with any obstacles or walls or if spaceship jumped outside of game boarders;
        # collidelist will return index if collision and -1 if not
        if player.rect.collidelist(self.comets.sprites()) > -1 or player.rect.collidelist(
                self.walls.sprites()) > -1 or player.rect.left < self.last_wall_pos[0] or player.rect.right > \
                self.last_wall_pos[1]:
            self.frames_with_collision += 1
        else:
            self.frames_with_collision = 0

        # check for collision threshold of consecutive frames with collision
        if self.frames_with_collision > self.frames_collision_threshold:
            player.crashed = True

    def check_for_drift(self):
        player = self.player.sprite
        self.drift.x = 0

        for sprite in self.drift_tiles.sprites():
            if player.rect.top in range(sprite.rect.top, sprite.rect.bottom):
                self.drift.x = -sprite.direction
            elif player.rect.bottom in range(sprite.rect.top, sprite.rect.bottom):
                self.drift.x = -sprite.direction

    def get_soc_response(self):
        self.SoC = self.agent.HL_SoC
        return self.SoC

    def get_prior_response(self):
        self.prior = self.agent.HL_SoC

    def get_data(self, scaling):

        frame_data = pd.DataFrame(columns=self.columns)

        player = self.player.sprite
        frame_data.at[0, 'player_pos'] = [player.rect.x, player.rect.y]  # player position will stay the same throughout
        frame_data.collision = player.crashed
        frame_data.current_input = self.current_input
        frame_data.drift_enabled = self.drift_enabled
        frame_data.current_drift = self.drift.x
        frame_data.level_done = self.level_done

        frame_data.time_played = self.time_played
        frame_data.trial = self.trial
        frame_data.attempt = self.attempt
        frame_data.level_size_y = self.level_size_y

        # inserting only last wall tile (bottom right of level) for later reconstruction of complete walls
        last_wall_tile = self.walls.sprites()[-1]
        frame_data.at[0, 'last_walls_tile'] = [last_wall_tile.rect.x, last_wall_tile.rect.y]

        # obstacles
        frame_data.at[0, 'visible_obstacles'] = self.visible_obstacles

        # drift
        frame_data.at[0, 'visible_drift_tiles'] = self.visible_drift_tiles

        # action goal
        frame_data.action_goal_selected = self.action_goal_selected
        frame_data.action_goal_x = self.agent.action_goal[0]
        frame_data.action_goal_y = self.agent.action_goal[1]

        # append everything to pandas DataFrame
        self.data = pd.concat([self.data, frame_data], ignore_index=True)
        self.data.prior = self.prior
        self.data.SoC = self.SoC

    def run(self, time_played, player_position, scaling):

        self.time_played = time_played
        player = self.player.sprite

        # updating visible obstacles
        self.visible_obstacles = []
        for sprite in self.comets.sprites():
            if 0 <= sprite.rect.y <= (observation_space_size_y - bottom_edge) * scaling:
                self.visible_obstacles.append([sprite.rect.x, sprite.rect.y])

        # updating visible drift tiles
        self.visible_drift_tiles = []
        for sprite in self.drift_tiles.sprites():
            if 0 <= sprite.rect.y <= (observation_space_size_y - bottom_edge) * scaling:
                self.visible_drift_tiles.append([sprite.rect.x, sprite.rect.y])

        # updating adjacent wall tiles y pos (left wall, right wall)
        self.adjacent_wall_tiles_x_pos = [self.walls.sprites()[0].rect.x, self.walls.sprites()[1].rect.x]

        # check for level done: if player went over finish line => level_done
        finish_line = self.finish_line.sprites()[-1]
        if finish_line.rect.bottom < player.rect.top:  # (observation_space_size_y - bottom_edge) * scaling:
            if question_soc:
                # ask for SoC:
                display_soc_question(self.display_surface)
                response = self.get_soc_response()
                if response is not None:
                    self.level_done = True
                    self.quit = True
                    # write data of all frames to csv
                    self.get_data(scaling)
                    self.data.to_csv(f'data/{self.code}_output_{self.trial}_{self.n_run:0>2}.csv', sep=',', index=False)
            else:
                self.level_done = True
                self.quit = True
                # write data of all frames to csv
                self.get_data(scaling)
                self.data.to_csv(f'data/{self.code}_output_{self.trial}_{self.n_run:0>2}.csv', sep=',', index=False)

        elif player.crashed:
            if question_soc:
                # ask for SoC:
                display_soc_question(self.display_surface)
                response = self.get_soc_response()
                if response is not None:
                    if self.time_played > self.replay_threshold:
                        self.level_done = True
                    self.quit = True
                    # write data of all frames to csv
                    self.get_data(scaling)
                    self.data.to_csv(f'data/{self.code}_output_{self.trial}_{self.n_run:0>2}.csv', sep=',', index=False)
            else:
                if self.time_played > self.replay_threshold:
                    self.level_done = True
                self.quit = True
                # write data of all frames to csv
                self.get_data(scaling)
                self.data.to_csv(f'data/{self.code}_output_{self.trial}_{self.n_run:0>2}.csv', sep=',', index=False)

        else:
            self.level_done = False

            # ask for prior

            if self.prior:
                player.animate(self.current_input)

                if player.rect.y < player_position[1]:
                    player.approach(velocity, scaling)
                    pass
                if player.rect.y >= player_position[1]:
                    # update sprite positions
                    # update level tiles
                    self.comets.update(velocity, scaling, self.horizontal_movement)
                    self.walls.update(velocity, scaling, self.horizontal_movement)
                    self.drift_tiles.update(velocity, scaling, self.horizontal_movement)
                    self.particles.update(velocity, scaling, self.horizontal_movement)
                    self.finish_line.update(velocity, scaling, self.horizontal_movement)

                    # update action goal
                    self.agent.update_action_goal(velocity, scaling, self.horizontal_movement)

                # check for collision
                self.check_for_collision()
                # check for drift
                self.check_for_drift()

                # draw sprites
                # draw comets and tiles
                self.particles.draw(self.display_surface)
                self.comets.draw(self.display_surface)
                self.walls.draw(self.display_surface)
                self.drift_tiles.draw(self.display_surface)
                self.finish_line.draw(self.display_surface)
                self.bottom_edge.draw(self.display_surface)
                # to display finish line when on screen but under bottom edge,
                # simply call draw method of finish_line.draw() AFTER buttom_edge.draw()

                # calling update here, because otherwise agent would get empty input in first frame...
                self.update()

                # draw agent
                self.player.draw(self.display_surface)

                # draw transparent circle around action goal
                draw_circle_alpha(surface=self.display_surface, color=(255, 0, 0, 100), center=self.agent.gaze_location,
                                  radius=degree_to_pixel(1))
                # draw fixated action goal
                pygame.draw.circle(self.display_surface, (255, 0, 0), self.agent.gaze_location, 2)
                # draw SoC indicators
                # low-level
                pygame.draw.rect(self.display_surface, (50, 168, 82),
                                 (player.rect.x + 2 * scaling, player.rect.y - self.agent.LL_SoC * 2 * scaling, scaling,
                                  self.agent.LL_SoC * 2 * scaling))
                # high-level
                pygame.draw.rect(self.display_surface, (232, 137, 12),
                                 (player.rect.x + 3.1 * scaling, player.rect.y - self.agent.HL_SoC * 2 * scaling,
                                  scaling, self.agent.HL_SoC * 2 * scaling))

                # draw keys
                if display_keys:
                    # right key
                    draw_rect_alpha(self.display_surface, (124, 252, 0, self.transparency_right), (160, 60, 90, 90))
                    draw_polygon_alpha(self.display_surface, (255, 255, 255, self.transparency_right),
                                       [(240, 105), (170, 70), (170, 140)])
                    # left key
                    draw_rect_alpha(self.display_surface, (124, 252, 0, self.transparency_left), (60, 60, 90, 90))
                    draw_polygon_alpha(self.display_surface, (255, 255, 255, self.transparency_left),
                                       [(70, 105), (140, 70), (140, 140)])

                self.get_data(scaling)

            elif not self.prior:
                # ask for prior belief
                display_prior_question(self.display_surface)
                self.get_prior_response()

        return self.quit, self.level_done
