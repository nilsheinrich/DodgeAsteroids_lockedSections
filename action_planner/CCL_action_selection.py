import numpy as np
import pandas as pd
import math
import random
import scipy.stats as st
import skimage.measure
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from action_planner.helper_functions import likelihood_function, normalized_posterior, bound
import sys
np.set_printoptions(threshold=sys.maxsize)


def pool_observation(observation_in_pixel, convolutionGranularity, resample=False):
    """
    Observation in pixel is pooled into an number of kernels with the number of kernels/pools given by
    the free parameter convolutionGranularity. The higher the value, the more granular is the
    representation of the visual environment. Kernels are subsequently convolved for mean pixel activation.
    """

    if resample:
        convolutionGranularity = 90
    else:
        convolutionGranularity = convolutionGranularity

    ratio = observation_in_pixel.shape[0] / observation_in_pixel.shape[1]  # ratio of rows to values in rows

    number_horizontal_strides = math.ceil(np.sqrt(convolutionGranularity / ratio))
    number_vertical_strides = math.ceil(convolutionGranularity / number_horizontal_strides)

    kernel_size_x = math.ceil(np.shape(observation_in_pixel)[1] / number_horizontal_strides)
    kernel_size_y = math.ceil(np.shape(observation_in_pixel)[0] / number_vertical_strides)

    pooled_observation = skimage.measure.block_reduce(observation_in_pixel, (kernel_size_y, kernel_size_x), np.mean)
    #print(f"pooled_observation: {pooled_observation}, "
    #      f"dimensions={len(pooled_observation[0])}*{len(pooled_observation)}")

    return kernel_size_x, kernel_size_y, pooled_observation, number_horizontal_strides, number_vertical_strides


def convolve_observation(observation_in_pixel, convolutionGranularity, min_percentage_for_rejection,
                         drift_situation=False, resample=False):
    """
    Every kernel of mean observation in pixel from pooled_observation is compared with
    min_percentage_for_rejection. If mean activation in kernel is higher than min_percentage_for_rejection, then
    1 is associated with kernel, reflecting the rejection of this specific possible action goal.
    """

    kernel_size_x, kernel_size_y, pooled_observation, number_horizontal_strides, number_vertical_strides = \
        pool_observation(observation_in_pixel, convolutionGranularity, resample=resample)

    # identify rejected action possibilities
    rejected_action_possibilities = list(zip(*np.where(pooled_observation > min_percentage_for_rejection)))
    action_field = list(zip(*np.where(pooled_observation < min_percentage_for_rejection)))

    if not drift_situation:
        # delete each possible in possibles that shares second digit with one in rejected and
        # the first digit of which is larger
        rejects = []
        for reject in rejected_action_possibilities:
            # rejects contains all goals on the same horizontal position as rejected_action_possibilities,
            # but vertically below rejected_action_possibilities
            rejects_ = [i for i in action_field if i[1] == reject[1] and i[0] > reject[0]]
            rejects = rejects + rejects_

        # only keep elements in action_field that are NOT in rejects
        action_field = [i for i in action_field if i not in rejects]
        rejected_action_possibilities = rejected_action_possibilities + rejects

    # also passing pooled observation
    pooled_observation = (pooled_observation > min_percentage_for_rejection).astype(int)  # cells only have 0 or 1

    # print(f"action field: {action_field}; rejected action possibilities: {rejected_action_possibilities}")

    return kernel_size_x, kernel_size_y, action_field, number_horizontal_strides, number_vertical_strides, rejected_action_possibilities, pooled_observation


def select_action_goal(PAR: dict, HL_SoC: float, observation_in_pixel, reference: tuple,
                       agent_pos_x: int, min_percentage_for_rejection: float, debug=False):
    """
    Possible action goals are received from generate_action_field. Possible action goals with mean activation set to 1
    are rejected and only those that are below 1 are considered. Then only those in row N are given to action goal
    selection process.

    N row from which possible action goals are considered is dependent on HL_SoC: loc variable.

    Of those remaining the one that is closest on the horizontal axis to the agent_pos_x is chosen.
    """

    kernel_size_x, kernel_size_y, action_field, number_horizontal_strides, number_vertical_strides, time, rejected_action_possibilities, rejects, _ = convolve_observation(PAR, observation_in_pixel, min_percentage_for_rejection)

    # the higher the HL_SoC, the higher the row number
    loc = int(HL_SoC*number_vertical_strides - len(rejected_action_possibilities)*0.001)
    # 0.001 being estimate in experimental data; which is multiplied by the number of populated pools

    # lay a probability mass function over the vertical strides with highest probability density centered over loc
    # space=5: [0, 1, 2, 3, 4]; loc:centered above row, sigma:arbitrary
    decision_space_y_probs = likelihood_function(space=np.arange(number_vertical_strides), mu=loc, sigma=1)
    decision_space_y = random.choices(population=np.arange(number_vertical_strides), weights=decision_space_y_probs, k=1)

    action_possibilities = [action_possibility for action_possibility in action_field if
                            action_possibility[0] == decision_space_y]

    # resample action field with highest granularity when there are no action_possibilities in desired loc
    if len(action_possibilities) < 1:
        kernel_size_x, kernel_size_y, action_field, number_horizontal_strides, number_vertical_strides, time, rejected_action_possibilities, rejects, _ = convolve_observation(PAR, observation_in_pixel, min_percentage_for_rejection, resample=True)

        action_possibilities = [action_possibility for action_possibility in action_field if
                                action_possibility[0] == decision_space_y]

    if debug:
        print(f"loc: {loc}; decision_space_y_probs: {decision_space_y_probs}; decision_space_y: {decision_space_y}")
    # what if there are no possible action goals? Move to nearest kernel where activation lowest
    # if len(possible_action_goals) < 1:

    # transform coordinates so that they match actual observation in pixel (without reference)
    action_possibilities_in_pixel = [(x * kernel_size_x + (kernel_size_x / 2), y * kernel_size_y + (kernel_size_y / 2))
                                     for y, x in action_possibilities]

    # choose action goal from all possible action goals by heuristic of closest on horizontal axis
    # (least amount of exerted action control)
    # print(f"possible action - convolved:{action_possibilities_in_pixel}; in pixel:{action_possibilities_in_pixel}")
    # [(44.5, y), (133.5, y), (222.5, y), (311.5, y), (400.5, y), (489.5, y)]
    """
    TODO: introduce minimization-maximization integration
    the action control to be exerted is kept minimized, but the distance to obstacles is kept maximized.
    How to: minimization is implemented below; maximization is easy, but how to integrate: centre point of both?
    There might even be the dynamic that human increase the distance kept towards obstacles when they sense a 
    decrease in HL SoC. We found an increase in the distance to the closest obstacle based on the number of visible
    drift tiles.
    """

    # minimizing the amount of control to-be-exerted
    action_goal = list(
        min(action_possibilities_in_pixel, key=lambda point: abs(point[0] - (agent_pos_x - reference[0]))))
    # agent_pos_x still in pixel coordinates, therefore subtracting reference

    # get column of action goal for monitoring purposes
    action_goal_col = int((action_goal[0] - (kernel_size_y / 2)) / kernel_size_y)

    # find coordinates where the value is
    # =0, for free space
    # =1, for populated space
    zero_coords = np.argwhere(observation_in_pixel == 1)

    # extract x and y coordinates
    x_coords = zero_coords[:, 1]
    y_coords = zero_coords[:, 0]

    activations = pd.DataFrame({'x_coords': x_coords, 'y_coords': y_coords})

    # kernel density
    x_density = st.gaussian_kde(activations.x_coords)
    y_density = st.gaussian_kde(activations.y_coords)
    # axis objects
    x_axis = np.arange(1, observation_in_pixel.shape[1]+1, 1)
    y_axis = np.arange(1, observation_in_pixel.shape[0]+1, 1)

    # bottom-up saliency map
    saliency_map_x = x_density.evaluate(x_axis)
    saliency_map_y = y_density.evaluate(y_axis)

    # top-down action goal (acuity dependent on representation acuity/granularity =kernel_size)
    """
    how to choose sigma? 95% of the probability density should be within the kernel (kernel_size/2)
    ==> kernel_size_/4?
    """
    action_goal_x_normalized = likelihood_function(space=x_axis, mu=action_goal[0], sigma=kernel_size_x/4)
    action_goal_y_normalized = likelihood_function(space=y_axis, mu=action_goal[1], sigma=kernel_size_y/4)

    # bayesian integration of saliency map and action goal
    posterior_x = normalized_posterior(prior=saliency_map_x, likelihood=action_goal_x_normalized)
    posterior_y = normalized_posterior(prior=saliency_map_y, likelihood=action_goal_y_normalized)

    # location of highest activation is selected location; adding reference for pixel coordinates
    highest_activation_x = pd.Series(posterior_x).idxmax() + reference[0]
    highest_activation_y = pd.Series(posterior_y).idxmax() + reference[1]

    ############################################
    if debug:
        # store where there is populated space, meaning there are obstacles
        populated_space_zero_coords = np.argwhere(observation_in_pixel == 1)
        populated_space_x_coords = populated_space_zero_coords[:, 1]
        populated_space_y_coords = populated_space_zero_coords[:, 0]
        populated_space_activations = pd.DataFrame({'x_coords': populated_space_x_coords,
                                                    'y_coords': populated_space_y_coords})

        # Plotting action field
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        plt.style.use("dark_background")

        ax = sns.jointplot(data=populated_space_activations, x="x_coords", y="y_coords", color="w", space=0)

        for horizontal_stride in range(1, number_horizontal_strides+1):
            ax.ax_joint.axvline(kernel_size_x * horizontal_stride, c="r")

        for vertical_stride in range(1, number_vertical_strides+1):
            ax.ax_joint.axhline(kernel_size_y * vertical_stride, c="r")

        # draw cognitive (top-down) action goal
        ax.ax_joint.axvline(action_goal[0], c="blue")
        ax.ax_joint.axhline(action_goal[1], c="blue")

        # draw integrated action goal (bottom-up + top-down)
        ax.ax_joint.axvline(highest_activation_x-reference[0], c="green")
        ax.ax_joint.axhline(highest_activation_y-reference[1], c="green")

        # mark rejected
        rejected = rejected_action_possibilities
        if rejected is not None:
            # print(f'rejected: {rejected}')
            rejected = [(x * kernel_size_x,
                         y * kernel_size_y)
                        for y, x in rejected]

            x_val = [x[0] for x in rejected]
            y_val = [y[1] for y in rejected]
            # print(x_val, y_val)

            for x_coord, y_coord in zip(x_val, y_val):
                rect = patches.Rectangle((x_coord, y_coord), kernel_size_x, kernel_size_y, linewidth=1, edgecolor='r',
                                         facecolor='r', alpha=0.4)
                ax.ax_joint.add_patch(rect)

        ax.ax_joint.get_xaxis().set_visible(False)
        ax.ax_joint.get_yaxis().set_visible(False)

        ax.ax_marg_x.set_xlim(0, observation_in_pixel.shape[1])
        ax.ax_marg_y.set_ylim(0, observation_in_pixel.shape[0])

        ax.fig.axes[0].invert_yaxis()

        plt.savefig('plots/action_field.png')
        plt.close('all')
    ############################################

    # assess time taken for action selection
    action_selection_time = np.random.uniform(60, 110, 1)
    time += action_selection_time

    # boost HL_SoC for new action goal chosen
    HL_SoC += PAR["SoCBoost"]
    HL_SoC = bound(0, 1, HL_SoC)  # HL_SoC bottoms at 0.0 and tops at 1.0

    return [highest_activation_x, highest_activation_y], action_goal_col, time, HL_SoC


def calc_spread(pooled_observation):
    # Get positions of 1s
    ones_positions = np.argwhere(pooled_observation == 1)

    if ones_positions.size == 0:
        return None  # No 1s found, return None or handle separately

    # Compute mean and standard deviation
    mean_position = np.mean(ones_positions, axis=0)
    std_deviation = np.std(ones_positions, axis=0)

    # Compute bounding box
    min_y, min_x = np.min(ones_positions, axis=0)
    max_y, max_x = np.max(ones_positions, axis=0)
    bounding_box_area = (max_x - min_x + 1) * (max_y - min_y + 1)

    # Compute density
    density = len(ones_positions) / bounding_box_area if bounding_box_area > 0 else 0

    # Compute mean pairwise distance using scipy's optimized pdist
    pairwise_distances = distance.pdist(ones_positions, metric='euclidean')
    mean_pairwise_distance = np.mean(pairwise_distances) if pairwise_distances.size > 0 else 0

    return mean_position, std_deviation, bounding_box_area, density, mean_pairwise_distance


def select_drift_path(PAR: dict, x_pos, vertical_dist, observation_in_pixel, SoC, drift_prior, drift_direction,
                      min_percentage_for_rejection: float, drift_situation=True, debug=False):
    """
    ...
    """
    if debug:
        np.savetxt("observation.csv", observation_in_pixel, delimiter=",")
    dx = drift_prior * drift_direction
    dy = observation_in_pixel.shape[0]  # dx & dy need to be in pixel scale
    slope = dx / dy  # expected trajectory in pixel scale

    # convolutionGranularity = PAR["convolutionGranularity"]

    kernel_size_x, kernel_size_y, action_field, number_horizontal_strides, number_vertical_strides, rejected_action_possibilities, pooled_observation = \
        convolve_observation(observation_in_pixel, PAR["convolutionGranularity"], min_percentage_for_rejection, drift_situation=drift_situation, resample=False)

    # ACT-R production
    # dividing pooled_observation
    num_cols = pooled_observation.shape[1]

    num_sections = 3  # in how many sections is the situation mentally divided
    # compute section sizes
    left_size = num_cols // num_sections
    middle_size = num_cols // num_sections + num_cols % num_sections  # gets the remainder
    right_size = num_cols // num_sections

    # Split each row accordingly
    left_section = pooled_observation[:, :left_size]
    middle_section = pooled_observation[:, left_size:left_size + middle_size]
    right_section = pooled_observation[:, left_size + middle_size:]

    hratio_ = x_pos / observation_in_pixel.shape[1]
    if hratio_ < 1/3:
        left_effort, middle_effort, right_effort = 0, 1, 2
    elif hratio_ > 2/3:
        left_effort, middle_effort, right_effort = 2, 1, 0
    else:
        left_effort, middle_effort, right_effort = 1, 0, 1

    left_risk, middle_risk, right_risk = len(np.argwhere(left_section == 1)), len(np.argwhere(middle_section == 1)), len(np.argwhere(right_section == 1))
    ##################

    # ACT-R production
    if drift_direction > 0:
        right_risk += 5  # arbitrarily chosen value
    elif drift_direction < 0:
        left_risk += 5
    ##################

    # ACT-R production
    # Top-down decision
    W_risk = 1.0
    W_effort = 1.0

    if SoC < PAR['SoCWeightingThreshold']:
        # risk weighted higher
        W_risk += 1.0
    else:  # SoC >= PAR['SoCWeightingThreshold']
        # effort weighted higher
        W_effort += 1.0

    # compute decision weight for sections
    decisionValue_left = W_risk*left_risk + W_effort*left_effort
    decisionValue_middle = W_risk * middle_risk + W_effort * middle_effort
    decisionValue_right = W_risk * right_risk + W_effort * right_effort

    # Identify section with lowest decision value
    sections = pd.DataFrame(np.array([[0, left_section.shape[1], left_risk, left_effort, decisionValue_left],
                                      [left_size, middle_section.shape[1], middle_risk, middle_effort, decisionValue_middle],
                                      [left_size+middle_size, right_section.shape[1], right_risk, right_effort, decisionValue_right]]),
                            columns=['start_x', 'width', 'risk', 'effort', 'decisionValue'])
    chosen_section = sections[sections.decisionValue == sections.decisionValue.min()]
    if len(chosen_section) > 1:  # it might be that the decisionValue is the same for several sections
        chosen_section = chosen_section.sample(n=1)

    # min_x_start, max_x_start = np.array(chosen_section.start_x), np.array(chosen_section.start_x+chosen_section.width)
    ##################

    # get board dimensions
    height, width = pooled_observation.shape

    # induce more distance to bottom walls
    # pooled_observation[-1, 0] = 1
    # pooled_observation[-1, -1] = 1

    # ACT-R production
    if dx > 0:  # positive slope
        min_x_start, max_x_start = 1, pooled_observation.shape[1] - abs(slope * observation_in_pixel.shape[0] / kernel_size_x)
    else:  # negative slope
        min_x_start, max_x_start = abs(slope * observation_in_pixel.shape[0] / kernel_size_x)+1, pooled_observation.shape[1] -1
    ##################

    # and within dynamic range of x_pos to guarantee that agent makes it to position.
    # dynamic range is bound to vertical distance to drift section.
    # (later for effort)
    agent_pooled_x_pos = math.floor(x_pos / kernel_size_x)
    pooled_vertical_dist = vertical_dist / kernel_size_x
    # unintuitive but it's about how far I can still steer horizontally given the vertical distance

    lower_bound = agent_pooled_x_pos - pooled_vertical_dist
    upper_bound = agent_pooled_x_pos + pooled_vertical_dist

    # print(f"min:{min_x_start}, max:{max_x_start}; bounds:{lower_bound}, {upper_bound}")
    if min_x_start < lower_bound:
        min_x_start = math.ceil(lower_bound)
        # print("adjusted lower bound")
    elif max_x_start > upper_bound:
        max_x_start = math.floor(upper_bound)
        # print("adjusted upper bound")

    # identifying potential starting positions
    candidate_xs = np.arange(min_x_start, max_x_start + 1)

    # select and store the best starting position and max smallest distance
    best_start_x = None
    expected_trajectory = None
    best_min_distance = -np.inf

    # all positions within grid that are =1
    ones_positions = np.argwhere(pooled_observation == 1)
    # print(f"populated kernels: {ones_positions}")

    # iterate over possible starting positions
    for start_x in candidate_xs:
        # simulate trajectory
        trajectory = []
        x, y = start_x, -1  # start above the first row

        while y < height - 1:  # stop before exceeding the board
            y += 1  # move down
            x += slope  # move left or right
            trajectory.append((y, x))

        trajectory = np.array(trajectory)
        # print(trajectory)

        # Compute the distance from trajectory points to ones_positions
        distances = distance.cdist(trajectory, ones_positions, metric='euclidean')

        # Find the minimum distance to any "1"
        min_distance = np.min(distances)

        # Update best position if this one is safer
        if min_distance > best_min_distance:
            best_min_distance = min_distance
            best_start_x = start_x
            expected_trajectory = trajectory

    hratio = best_start_x/observation_in_pixel.shape[1]

    # plotting
    if debug:
        # store where there is populated space, meaning there are obstacles
        populated_space_zero_coords = np.argwhere(observation_in_pixel == 1)
        populated_space_x_coords = populated_space_zero_coords[:, 1]
        populated_space_y_coords = populated_space_zero_coords[:, 0]
        populated_space_activations = pd.DataFrame({'x_coords': populated_space_x_coords,
                                                    'y_coords': populated_space_y_coords})

        # Plotting action field
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))

        plt.style.use("dark_background")

        ax = sns.jointplot(data=populated_space_activations, x="x_coords", y="y_coords", color="w", space=0)

        for horizontal_stride in range(1, number_horizontal_strides+1):
            ax.ax_joint.axvline(kernel_size_x * horizontal_stride, c="r")

        for vertical_stride in range(1, number_vertical_strides+1):
            ax.ax_joint.axhline(kernel_size_y * vertical_stride, c="r")

        # mark rejected
        rejected = rejected_action_possibilities
        if rejected is not None:
            rejected = [(x * kernel_size_x,
                         y * kernel_size_y)
                        for y, x in rejected]

            x_val = [x[0] for x in rejected]
            y_val = [y[1] for y in rejected]
            # print(x_val, y_val)

            for x_coord, y_coord in zip(x_val, y_val):
                rect = patches.Rectangle((x_coord, y_coord), kernel_size_x, kernel_size_y, linewidth=1, edgecolor='r',
                                         facecolor='r', alpha=0.4)
                ax.ax_joint.add_patch(rect)

        scaled_expected_trajectory = [np.array([x * kernel_size_x + kernel_size_x/2, y * kernel_size_y]) for y, x in expected_trajectory]
        for point in scaled_expected_trajectory:
            # for point in expected_trajectory:
            ax.ax_joint.scatter(point[0], point[1], color='lime', s=100)

        ax.ax_joint.get_xaxis().set_visible(False)
        ax.ax_joint.get_yaxis().set_visible(False)

        ax.ax_marg_x.set_xlim(0, observation_in_pixel.shape[1])
        ax.ax_marg_y.set_ylim(0, observation_in_pixel.shape[0])

        ax.fig.axes[0].invert_yaxis()

        plt.savefig('plots/drift_situation.png')
        plt.close('all')

    return best_start_x, expected_trajectory, slope, kernel_size_x, kernel_size_y
