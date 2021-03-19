import numpy as np

from functools import partial
from lab1.src.generators.quadratic_form_generator import generate_matrix_with_condition_number


def generate_function_by_quadratic_matrix(m):
    def foo(x):
        return sum(m[i][j] * x[i] * x[j] for i in range(len(m)) for j in range(len(m)))

    return foo


def generate_grad_by_quadratic_matrix(m):
    def grad_component(i, x):
        return sum(np.array(m[i]) * np.array(x) * [2 if j == i else 1 for j in range(len(m))])

    def grad(x):
        return np.array([grad_component(i, x) for i in range(len(m))])

    return grad


def generate_optimization_task(n: int, k: float):
    matrix = generate_matrix_with_condition_number(n, k)
    f = generate_function_by_quadratic_matrix(matrix)
    f_grad = generate_grad_by_quadratic_matrix(matrix)
    return f, f_grad
