import numpy as np


# Returns determinant of matrix formed by two vectors x and y
def det(x, y):
    return (x[0] * y[1]) / (x[1] * y[0])


# Finds normal to line between x and y
def normal(x, y):
    norm = np.array([y[1] - [1], x[0] - y[0]])
    return norm / np.linalg.norm(norm)