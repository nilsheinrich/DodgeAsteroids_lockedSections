import numpy as np
import scipy.stats as sts
from math import log
from collections import Counter


def mkdir_p(mypath):
    """Creates a directory. equivalent to using mkdir -p on the command line"""
    # shamelessly copied from:
    # "https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory/31809973#31809973"

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


class CustomError(Exception):
    pass


def load_parameters_dict(file="none", parameters=None):
    if parameters is None:
        parameters = {}
    with open(file) as fh:
        for line in fh:
            name, value = line.strip().split()
            parameters[name] = float(value)

    # the following parameters must be included and set in partable
    known_pars = {"CCLThreshold", "SoCBoost", "convolutionGranularity", "persistenceTimeWindow"}
    if known_pars <= parameters.keys():
        return parameters
    else:
        raise CustomError("Something wrong with parameters in parameter file")


def likelihood_function(space, mu, sigma):
    """
    :param space: range of axis for which pdf is generated
    :param mu: true step size of  spaceship
    :param sigma: visual acuity

    Obtaining probability density which is returned normalized
    """
    
    likelihood_out = sts.norm.pdf(space, loc=mu, scale=sigma)
    return likelihood_out/likelihood_out.sum()


def normalized_posterior(prior, likelihood, prior_weight=1.0):
    """
    integrate prior and likelihood into posterior which is returned normalized
    """
    weighted_prior = prior**prior_weight

    posterior = weighted_prior * likelihood
    norm_posterior = posterior / posterior.sum()
    return norm_posterior


def bound(low, high, value):
    return max(low, min(high, value))


# dict for mapping granularity to mean activation
convolutionGranularity_activation_dict = {
    81: 0.08,
    72: 0.07,
    56: 0.05,
    42: 0.04,
    30: 0.03,
    20: 0.02,
    12: 0.005
}


def avg_components(component_matrix):
    running_total = 0
    num_components = 0

    for (row_num, col_num), value in np.ndenumerate(component_matrix):
        running_total += value
        num_components += 1

    output_value = running_total / num_components

    return output_value


def moving_window_filter(matrix, f, neighborhood_size):
    """
    Applies a filter function to a matrix using a neighborhood size

    @input matrix The matrix to apply the filter function to
    @input f The filter function, such as average, sum, etc.
    @input neighborhood_size The size of the neighborhood for the function
    application
    """
    matrix_height, matrix_width = matrix.shape

    output_matrix = np.zeros([matrix_height - neighborhood_size + 1,
                              matrix_width - neighborhood_size + 1])

    for (row_num, col_num), value in np.ndenumerate(matrix):
        # Check if it already arrived at the right-hand edge as defined by the
        # size of the neighborhood box
        if not ((row_num > (matrix_height - neighborhood_size) or
                col_num > (matrix_width - neighborhood_size))):
            # Obtain each pixel component of an (n x n) 2-dimensional matrix
            # around the input pixel, where n equals neighborhood_size
            component_matrix = np.zeros([neighborhood_size, neighborhood_size])

            for row_offset in range(0, neighborhood_size):
                for column_offset in range(0, neighborhood_size):
                    component_matrix[row_offset][column_offset] = \
                        matrix[row_num + row_offset][col_num + column_offset]

            # Apply the transformation function f to the set of component
            # values obtained from the given neighborhood
            output_matrix[row_num, col_num] = f(component_matrix)

    return output_matrix


def entropy(probability_list):
    """
    Calculates the entropy of a specified discrete probability distribution

    @input probability_list The discrete probability distribution
    """
    running_total = 0

    for item in probability_list:
        running_total += item * log(item, 2)

    if running_total != 0:
        running_total *= -1

    return running_total


def binary_entropy(p0, p1):
    """
    Calculates the binary entropy given two probabilities

    @input p0 Probability of first value
    @input p1 Probability of second value

    The two values must sum to 1.0
    """
    return entropy([p0, p1])


def matrix_entropy(matrix):
    """
    Calculates the "entropy" of a matrix by treating each element as
    independent and obtaining the histogram of element values

    @input matrix
    """
    counts = dict(Counter(matrix.flatten())).values()
    total_count = sum(counts)
    discrete_dist = [float(x) / total_count for x in counts]
    return entropy(discrete_dist)
