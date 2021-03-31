import math
import numpy as np

from lab2.src.methods.conjugate_method import (
    conjugate_direction_method,
    conjugate_gradient_method
)

from lab2.src.methods.newton_method import newton_method


def quadratic_function(x: np.ndarray):
    return 100 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2


def quadratic_function_grad(x: np.ndarray):
    return np.array(
        [
            200 * x[0] - 200 * x[1] - 2,
            200 * (x[1] - x[0])
        ], dtype="float64"
    )


def quadratic_function_hess(x: np.ndarray):
    return np.array(
        [
            [202, -200],
            [-200, 200]
        ], dtype="float64"
    )


def quadratic_function_Q():
    return np.array(
        [
            [202, -200],
            [-200, 200]
        ], dtype="float64"
    )


def compare_quadratic():
    print("===Quadratic function===")
    result_directions = conjugate_direction_method(Q=quadratic_function_Q(),
                                                   b=np.array([-2, 0], dtype="float64"),
                                                   start=np.array([-1, -2], dtype="float64")
                                                   )
    result_conj_grad = conjugate_gradient_method(f=quadratic_function,
                                                 f_grad=quadratic_function_grad,
                                                 start=np.array([-1, -2], dtype="float64")
                                                 )
    result_newton = newton_method(f=quadratic_function,
                                  f_grad=quadratic_function_grad,
                                  f_hess=quadratic_function_hess,
                                  start=np.array([-1, -2], dtype="float64"))

    print(result_directions)
    print(result_conj_grad)
    print(result_newton)


#########################################

def rosenbrock_function(x: np.ndarray):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosenbrock_function_grad(x: np.ndarray):
    return np.array([
        400 * x[0] ** 3 + 2 * x[0] - 400 * x[0] * x[1] - 2,
        -200 * x[0] ** 2 + 200 * x[1],
    ], dtype="float64")


def rosenbrock_function_hess(x: np.ndarray):
    return np.array([
        [1200 * x[0] ** 2 + 2 - 400 * x[1], -400 * x[0]],
        [-400 * x[0], 200],
    ], dtype="float64")


def compare_rosenbrock_function():
    print("===Rosenbrock function===")
    result_conj_grad = conjugate_gradient_method(f=rosenbrock_function,
                                                 f_grad=rosenbrock_function_grad,
                                                 start=np.array([-1, -2], dtype="float64")
                                                 )
    result_newton = newton_method(f=rosenbrock_function,
                                  f_grad=rosenbrock_function_grad,
                                  f_hess=rosenbrock_function_hess,
                                  start=np.array([-1, -2], dtype="float64"))

    print(result_conj_grad)
    print(result_newton)


#########################################

def test_function(x: np.ndarray):
    return -2 * math.exp(-(((x[0] - 1) / 2) ** 2) - (x[1] - 1) ** 2) - \
           3 * math.exp(-(((x[0] - 2) / 3) ** 2) - ((x[1] - 3) / 2) ** 2)


def test_function_grad(x: np.ndarray):
    return np.array(
        [
            2 * (x[0] - 2) / 3
            * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
            + (x[0] - 1) * math.exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
            3 / 2 * (x[1] - 3)
            * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
            + 4 * (x[1] - 1) * math.exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
        ],
        dtype="float64",
    )


def test_function_hess(x: np.ndarray):
    return np.array(
        [
            [
                (-4 / 27 * (x[0] - 2) ** 2 + 2 / 3)
                * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
                + (1 - (x[0] - 1) ** 2 / 2)
                * math.exp(-((x[0] - 1) ** 2) / 4 - (x[1] - 1) ** 2),
                -1 / 3 * (x[0] - 2) * (x[1] - 3)
                * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
                - 2 * (x[0] - 1) * (x[1] - 1)
                * math.exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
            ],
            [
                -1 / 3 * (x[0] - 2) * (x[1] - 3)
                * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
                - 2 * (x[0] - 1) * (x[1] - 1)
                * math.exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
                (-3 / 4 * (x[1] - 3) ** 2 + 3 / 2)
                * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
                + (4 - 8 * (x[1] - 1) ** 2)
                * math.exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
            ],
        ],
        dtype="float64",
    )


def compare_test_function():
    print("===Test function===")
    result_conj_grad = conjugate_gradient_method(f=test_function,
                                                 f_grad=test_function_grad,
                                                 start=np.array([1.5, 1.5], dtype="float64")
                                                 )
    result_newton = newton_method(f=test_function,
                                  f_grad=test_function_grad,
                                  f_hess=test_function_hess,
                                  start=np.array([1.5, 1.5], dtype="float64"))

    print(result_conj_grad)
    print(result_newton)


#########################################

def compare():
    compare_quadratic()
    compare_rosenbrock_function()
    compare_test_function()


if __name__ == '__main__':
    compare()
