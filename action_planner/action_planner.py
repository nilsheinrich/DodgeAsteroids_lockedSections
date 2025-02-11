import time
import numpy as np
import scipy.stats as st
from action_planner.helper_functions import likelihood_function, normalized_posterior, bound, \
    convolutionGranularity_activation_dict, matrix_entropy
from action_planner.CCL_action_selection import pool_observation


class ActionPlanner:
    def __init__(self, free_parameters, initial_position_x, observation_space_in_pixel):
        self.parameters = free_parameters

        # begin counting from here
        self.time_stamp = time.time()

        # search dict for appropriate min_percentage_for_rejection
        self.min_percentage_for_rejection = convolutionGranularity_activation_dict[
            self.parameters['convolutionGranularity']]

        # sense of control for SCL (LL) & CCL (HL)
        self.LL_SoC = 0.5
        self.HL_SoC = 0.5

        # observation in pixel
        self.observation_space_x_in_pixel = np.linspace(0, observation_space_in_pixel[0],
                                                        num=observation_space_in_pixel[0])
        self.observation_space_y_in_pixel = observation_space_in_pixel[1]
        # assessed complexity of current situation
        self.instance_complexity = 0.0

        self.agent_pos_x = initial_position_x  # agent will always stay in screen center

        # visual acuity for matching visually perceived stimuli and intended locations
        self.visual_acuity = 3  # arbitrarily chosen

        uniform_dist = st.uniform.pdf(self.observation_space_x_in_pixel) + 1
        self.step_size_prior = uniform_dist / uniform_dist.sum()  # normalized prior

        # priors for imposed movement by drift (separate from step size) - over whole drift section
        self.drift_prior = likelihood_function(space=self.observation_space_x_in_pixel,
                                               mu=230,  # true value
                                               sigma=50)  # well informed prior
        self.drift_applying = False  # True vs. False: inferred state (from visual feedback) of drift applying or not
        # drift can be on screen but not applying. This variable only refers to the model inferring whether drift
        # applies not whether it is visible.
        self.drift_direction = None

        # horizontal movement either by own action or by drift
        self.perceived_step_size = None  # likelihood

        # history of prediction errors
        self.prediction_errors = []
        # threshold for when PEs are actually considered as PE (decrease LL_SoC)
        self.PE_threshold = 0.00288  # 0.0001, in case of integration before PE ident

        # memory content
        self.declarative_memory = {
            # dict of dicts
            # exemplary_chunk: {
            #   'observation':
            #   'SoC':
            #   'actionGoal':
            #   'utility':
            #   }
        }
        # working memory
        self.situated_state = {
            #   'observation':
            #   'HL_SoC':
            #   'action_field':
            #   'actionGoal':
        }

        # trace of states
        self.traced_states = []
        # initialize (empty) list of active processes
        self.active_processes = []

        self.action_goal = None
        self.action_goal_col = None
        # when is action goal reached (not necessarily if point is met)
        self.target_radius = 5  # radius around action goals that is deemed as sufficiently for action_goal_reached
        # is action goal successfully reached
        self.action_goal_reached = False  # True vs. False

        self.gaze_location = None

        # oculomotor control
        self.executing_saccade = False  # True vs. False

        # currently executed actions; driven by SCL
        self.action = None  # can be None vs. 'Right' vs. 'Left'

    def update_action_goal(self, speed, scaling, horizontal_movement):
        """
        Due to the environment moving around the action planner, the action goal has to be updated in every time step
        """
        # oculomotor error (nystagmus - small shifts of fovea while maintaining fixation)
        # maybe introduce center bias here
        # offset_x = np.random.randint(-1, 2, 1)[0]  # 2 because exclusive
        # offset_y = np.random.randint(-1, 2, 1)[0]
        # vertical movement
        # self.action_goal[1] -= ((2/3 * scaling) * speed) + offset_y
        self.action_goal[1] -= (scaling * speed)  # + offset_y
        # horizontal movement
        self.action_goal[0] += (horizontal_movement * scaling * speed)  # + offset_x

    def assess_action_goal(self, observation_in_pixel, reference, radius=12):  # radius that is =5° visual angle?
        """
        This function reflects the mental assessment of the situation. The if conditions reflect the conclusions the
        agent might draw. The following conclusions can be drawn:

        - about the complexity of the instance. The ratio of populated kernels to overall kernels is simply taken as
        the complexity of the instance. It will directly affect HL SoC ("I don't feel in control of this situation").

        - whether the action planner realized its plan (reached the action goal to an appropriate degree reflected by
        radius given in pixel). The radius reflects the visual degrees the fovea spans. If the agent enters foveal
        vision, this will be assessed as the action goal being reached.
        """
        # complexity of instance
        kernel_size_x, _, pooled_observation, number_horizontal_strides, _ = pool_observation(self.parameters, observation_in_pixel)
        binary_observation = (np.array(pooled_observation) > self.min_percentage_for_rejection).astype(int)
        new_complexity = matrix_entropy(binary_observation)

        # did complexity change? if yes, it affects HL_SoC
        change_in_complexity = self.instance_complexity - new_complexity
        self.HL_SoC += change_in_complexity
        self.HL_SoC = bound(0, 1, self.HL_SoC)  # HL_SoC bottoms at 0.0 and tops at 1.0

        # update instance complexity
        self.instance_complexity = new_complexity

        # for self.action_goal[1], the vertical range, it is sufficient for the action goal to enter specific region in
        # front of action planner agent
        if (self.agent_pos_x - radius <= self.action_goal[0] <= self.agent_pos_x + radius) & (
                self.action_goal[1] <= 236 + 100):
            # 236 being player.sprite.rect.bottom after approach at the start of level
            self.action_goal_reached = True
            self.action_goal = None
        else:
            # check if action goal is in same horizontal position as an obstacle
            populated_kernels = list(zip(*np.where(pooled_observation > self.min_percentage_for_rejection)))
            action_goal_x = self.action_goal[0]-reference[0]
            for populated_kernel in populated_kernels:
                if (populated_kernel[1]*60) < action_goal_x < (populated_kernel[1]*60+kernel_size_x):
                    # if the action goal at t-1 was also abandoned, then:
                    # self.HL_SoC -= self.HL_SoC/5
                    # self.HL_SoC = bound(0, 1, self.HL_SoC)
                    self.action_goal = None

    def apply_motor_control(self):
        """
        regulatory control on sensorimotor control layer
        """
        if self.agent_pos_x > self.action_goal[0] + self.target_radius:
            self.action = 'Left'
        elif self.agent_pos_x < self.action_goal[0] - self.target_radius:
            self.action = 'Right'
        else:
            self.action = None

    def prediction_error(self):
        """
        The agent will always stay at the same position (centered) and because of this what is actually inferred is
        the horizontal movement of the environment.
        
        likelihood is generated based on true step size of environment incorporating visual acuity. 
        """
        likelihood = likelihood_function(space=self.observation_space_x_in_pixel,
                                         mu=self.perceived_step_size,
                                         sigma=self.visual_acuity)
        # posterior = normalized_posterior(prior=self.step_size_prior, likelihood=likelihood)
        # print(self.step_size_prior, posterior)
        # prediction_error = st.wasserstein_distance(self.step_size_prior, posterior)
        prediction_error = st.wasserstein_distance(self.step_size_prior, likelihood)
        # print(f"Wasserstein distance: {prediction_error}")
        # prediction_error = st.entropy(pk=self.step_size_prior, qk=posterior)
        # prediction_error = st.entropy(self.step_size_prior, likelihood)
        # print(f"KL divergence: {prediction_error}")

        # when prediction errors exceed threshold, then reduce LL_SoC
        if prediction_error > self.PE_threshold:
            LL_SoC_loss = prediction_error * 100  # 1000 when KL
            # arbitrary inflation of of prediction error for loss in SoC
            print(f"LL_SoC_loss: {LL_SoC_loss}")
            self.LL_SoC -= LL_SoC_loss
            self.LL_SoC = bound(0, 1, self.LL_SoC)  # LL_SoC bottoms at 0.0 and tops at 1.0
            # when LL_SoC exceeds threshold, then reduce HL_SoC
            if self.LL_SoC < self.parameters["CCLThreshold"]:
                # the higher HL_SoC, the greater the loss
                self.HL_SoC -= (self.HL_SoC / 5)
                # self.HL_SoC -= self.parameters["HLSoCLoss"]
                self.HL_SoC = bound(0, 1, self.HL_SoC)  # HL_SoC bottoms at 0.0 and tops at 1.0
        elif prediction_error <= self.PE_threshold:
            # if predictions came true, then boost LL SoC with Shannon entropy of prior distribution for step size
            """
            The boost in LL SoC should be higher if a more precise prediction comes true. Shannon 
            entropy increases with increasing standard deviation though...
            """
            self.LL_SoC += (1 / st.entropy(pk=self.step_size_prior)) / 10
            self.LL_SoC = bound(0, 1, self.LL_SoC)  # LL_SoC bottoms at 0.0 and tops at 1.0

        # update prior in self.step_size_prior with new posterior
        posterior = normalized_posterior(prior=self.step_size_prior, likelihood=likelihood)
        self.step_size_prior = posterior

        self.prediction_errors.append(prediction_error)  # storing prediction errors
        return prediction_error
