import numpy as np
import pandas as pd
import math
import random
import scipy.stats as st
import skimage.measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from action_planner.helper_functions import likelihood_function, normalized_posterior, bound
import sys
np.set_printoptions(threshold=sys.maxsize)


def pool_observation(PAR: dict, observation_in_pixel, resample=False):
    """
    Observation in pixel is pooled into an number of kernels with the number of kernels/pools given by
    the free parameter convolutionGranularity. The higher the value, the more granular is the
    representation of the visual environment. Kernels are subsequently convolved for mean pixel activation.

    In the future: make number of kernels/pools depend on HL_SoC. The higher the HL_SoC the higher the
    convolutionGranularity .i.e the smaller the kernel size?
    """
    # df = pd.DataFrame(data=observation_in_pixel[0:, 0:],
    #                  index=[i for i in range(observation_in_pixel.shape[0])],
    #                  columns=['f' + str(i) for i in range(observation_in_pixel.shape[1])])
    # df.to_csv("surfarray.csv", sep=',', index=False)
    # resampling? then highest granularity by default
    if resample:
        convolutionGranularity = 81
    else:
        convolutionGranularity = PAR["convolutionGranularity"]

    # pooling
    number_horizontal_strides = math.ceil(np.sqrt(convolutionGranularity))
    kernel_size_x = math.ceil(np.shape(observation_in_pixel)[1] / number_horizontal_strides)
    number_vertical_strides = int(convolutionGranularity / number_horizontal_strides)

    """
    There might be need for a dynamic granularity: when under the default granularity no valid action goal can be
    determined, granularity might be increased to search for free spaces in between pools that were populated before 
    under smaller granularity.
    """
    # the following code is for a dynamic granularity, in which the number of pools is directly influenced by HL_SoC
    # convolutionGranularity = HL_SoC_convolutionGranularity_dict[round(HL_SoC, 1)]
    # number_horizontal_strides = math.ceil(np.sqrt(convolutionGranularity))
    # kernel_size_x = math.ceil(np.shape(observation_in_pixel)[1] / number_horizontal_strides)
    # number_vertical_strides = convolutionGranularity / number_horizontal_strides

    kernel_size_y = math.ceil(np.shape(observation_in_pixel)[0] / number_vertical_strides)
    pooled_observation = skimage.measure.block_reduce(observation_in_pixel, (kernel_size_y, kernel_size_x), np.mean)
    # print(f"pooled_observation: {pooled_observation}, "
    #      f"dimensions={len(pooled_observation[0])}*{len(pooled_observation)}")
    return kernel_size_x, kernel_size_y, pooled_observation, number_horizontal_strides, number_vertical_strides


def convolve_observation(PAR: dict, observation_in_pixel, min_percentage_for_rejection, resample=False):
    """
    Every kernel of mean observation in pixel from pooled_observation is compared with
    min_percentage_for_rejection. If mean activation in kernel is higher than min_percentage_for_rejection, then
    1 is associated with kernel, reflecting the rejection of this specific possible action goal.
    """

    kernel_size_x, kernel_size_y, pooled_observation, number_horizontal_strides, number_vertical_strides = pool_observation(PAR, observation_in_pixel, resample=resample)

    # identify rejected action possibilities
    # print(f'activation in cells: \n{pooled_observation}')
    rejected_action_possibilities = list(zip(*np.where(pooled_observation > min_percentage_for_rejection)))
    # print(f"rejected_action_possibilities = {rejected_action_possibilities}")
    action_field = list(zip(*np.where(pooled_observation < min_percentage_for_rejection)))
    # print(f"action_field before filter = {action_field}")
    # delete each possible in possibles that shares second digit with one in rejected and
    # the first digit of which is larger
    rejects = []
    for reject in rejected_action_possibilities:
        # rejects contains all goals on the same horizontal position as rejected_action_possibilities,
        # but vertically below rejected_action_possibilities
        rejects = [i for i in action_field if i[1] == reject[1] and i[0] > reject[0]]
        # only keep elements in action_field that are NOT in rejects
        action_field = [i for i in action_field if i not in rejects]
    # print(f"action_field: {action_field}")

    # assess time taken for conscious broadcast
    broadcast_time = np.random.uniform(200, 280, 1)
    # print(f"action field: {action_field}; rejected action possibilities: {rejected_action_possibilities}")
    return kernel_size_x, kernel_size_y, action_field, number_horizontal_strides, number_vertical_strides, broadcast_time, rejected_action_possibilities, rejects


def select_action_goal(PAR: dict, HL_SoC: float, observation_in_pixel, reference: tuple,
                       agent_pos_x: int, min_percentage_for_rejection: float, debug=False):
    """
    Possible action goals are received from generate_action_field. Possible action goals with mean activation set to 1
    are rejected and only those that are below 1 are considered. Then only those in row N are given to action goal
    selection process.

    N row from which possible action goals are considered is dependent on HL_SoC: loc variable.

    Of those remaining the one that is closest on the horizontal axis to the agent_pos_x is chosen.
    """

    kernel_size_x, kernel_size_y, action_field, number_horizontal_strides, number_vertical_strides, time, rejected_action_possibilities, rejects = convolve_observation(PAR, observation_in_pixel, min_percentage_for_rejection)

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
        kernel_size_x, kernel_size_y, action_field, number_horizontal_strides, number_vertical_strides, time, rejected_action_possibilities, rejects = convolve_observation(PAR, observation_in_pixel, min_percentage_for_rejection, resample=True)

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


def sample_gaze_location(HL_SoC: float, observation_in_pixel, action_goal_x, action_goal_y,
                         reference: tuple, debug=False):

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
    # x_axis = np.arange(1, observation_in_pixel.shape[1] + 1, 1)
    # y_axis = np.arange(1, observation_in_pixel.shape[0] + 1, 1)
    x_axis = np.arange(28, observation_in_pixel.shape[1] - 28, 1)
    y_axis = np.arange(28, observation_in_pixel.shape[0] - 28, 1)

    # bottom-up saliency map
    saliency_map_x = x_density.evaluate(x_axis)
    saliency_map_y = y_density.evaluate(y_axis)

    inv_saliency_map_x = (1 - saliency_map_x) / (1 - saliency_map_x).sum()
    inv_saliency_map_y = (1 - saliency_map_y) / (1 - saliency_map_y).sum()

    action_goal_map_x = likelihood_function(space=x_axis, mu=action_goal_x-reference[0], sigma=80)
    action_goal_map_y = likelihood_function(space=y_axis, mu=action_goal_y-reference[1], sigma=80)

    # compute posterior for each axis individually while inflating agent_map (=task_focus)
    posterior_x = normalized_posterior(prior=1000**action_goal_map_x,
                                       likelihood=inv_saliency_map_x,
                                       prior_weight=1-HL_SoC)
    posterior_y = normalized_posterior(prior=1000**action_goal_map_y,
                                       likelihood=inv_saliency_map_y,
                                       prior_weight=1-HL_SoC)

    # sampling from posterior to obtain selected action goal
    # final_gaze_x = np.random.choice(len(posterior_x), p=posterior_x) + reference[0]
    # final_gaze_y = np.random.choice(len(posterior_y), p=posterior_y) + reference[1]

    # location of highest activation is selected as action goal; adding reference for pixel coordinates
    final_gaze_x = pd.Series(posterior_x).idxmax() + reference[0]
    final_gaze_y = pd.Series(posterior_y).idxmax() + reference[1]

    if debug:
        populated_space_activations = pd.DataFrame({'x_coords': x_coords,
                                                    'y_coords': y_coords})

        # Plotting action field
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        plt.style.use("dark_background")

        ax = sns.jointplot(data=populated_space_activations, x="x_coords", y="y_coords", color="w", space=0)

        # draw integrated action goal (bottom-up + top-down)
        ax.ax_joint.axvline(final_gaze_x-reference[0], c="green")
        ax.ax_joint.axhline(final_gaze_y-reference[1], c="green")

        ax.ax_joint.get_xaxis().set_visible(False)
        ax.ax_joint.get_yaxis().set_visible(False)

        ax.ax_marg_x.set_xlim(0, observation_in_pixel.shape[1])
        ax.ax_marg_y.set_ylim(0, observation_in_pixel.shape[0])

        ax.fig.axes[0].invert_yaxis()

        plt.savefig(f'plots/kerneled_vis_map_{HL_SoC}.png')
        plt.close('all')

    return [final_gaze_x, final_gaze_y]


def select_drift_path(PAR: dict, observation_in_pixel, min_percentage_for_rejection: float, debug=True):
    """
    Possible action goals are received from generate_action_field. Possible action goals with mean activation set to 1
    are rejected and only those that are below 1 are considered. Then only those in row N are given to action goal
    selection process.

    N row from which possible action goals are considered is dependent on HL_SoC: loc variable.

    Of those remaining the one that is closest on the horizontal axis to the agent_pos_x is chosen.
    """

    kernel_size_x, kernel_size_y, action_field, number_horizontal_strides, number_vertical_strides, time, rejected_action_possibilities, rejects = convolve_observation(PAR, observation_in_pixel, min_percentage_for_rejection)

    ############################################
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

        ax.ax_joint.get_xaxis().set_visible(False)
        ax.ax_joint.get_yaxis().set_visible(False)

        ax.ax_marg_x.set_xlim(0, observation_in_pixel.shape[1])
        ax.ax_marg_y.set_ylim(0, observation_in_pixel.shape[0])

        ax.fig.axes[0].invert_yaxis()

        plt.savefig('plots/drift_situation.png')
        plt.close('all')
    ############################################

    return None
