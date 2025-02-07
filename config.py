# This file contains static variables that won't change in a single run

# how "far" is the agent able to see
observation_space_size_x, observation_space_size_y = 40, 60.5  # other possible sizes [80, 60]

# environment related parameters
velocity = 1/3  # 1 / (FPS/20)
level_size_x = observation_space_size_x  # level_size_y determined by walls (last wall ends level)
wall_size = 1
drift_tile_size_x = 1
particle_sizes = (2/16, 3/16, 4/16)

# drift 
# drift_magnitude = 1/2  # in case of using old level csv's that do not account for magnitude and simply state 1
# mapping
# drift = 0: leftwards drift
# drift = 2: rightwards drift


# agent related parameters
agent_size_x, agent_size_y = 2, 2
pre_trial_steps = 20  # steps spaceship has to take to reach starting_position in experimental setup
# can range from 0 - observation_space_y/2

# upscaling for visualization on screen
scaling = 14  # 19  # 20

# edge size for screen beyond walls and bottom
edge = 31.5  # 28 perfect for large screen with scaling = 20
bottom_edge = 15  # leaving 45 for visible observation window
