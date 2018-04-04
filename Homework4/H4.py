import matplotlib.pyplot as plt
import numpy as np
import random
import math
from numpy.linalg import inv


def my_mahalanobis(u, v, VI):
    u = _validate_vector(u)
    v = _validate_vector(v)
    VI = np.atleast_2d(VI)
    delta = u - v
    m = np.dot(np.dot(delta, VI), delta)
    return np.sqrt(m)


def my_mahalanobis_not_sqrted(u, v, VI):
    u = _validate_vector(u)
    v = _validate_vector(v)
    VI = np.atleast_2d(VI)
    delta = u - v
    m = np.dot(np.dot(delta, VI), delta)
    return m


def calculate_eta(u, v):
    inversed_v = inv(v)
    sum = (1.0 + (assignation) ** 2 * my_mahalanobis_not_sqrted(u, v, inversed_v)) / (1.0 + (assignation) ** 2)
    return sum


def calculate_u(u, v):
    inversed_v = inv(v)
    distance = my_mahalanobis(u, v, inversed_v)
    result = 1.0 + (my_mahalonobis(u, v, inversed_v) / calculate_eta(u, v)) ** 2
    return result ** -1
