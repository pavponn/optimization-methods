import math
import numpy as np
import pandas as pd

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
            202 * x[0] - 2 - 200 * x[1],
            -200 * x[0] + 200 * x[1]
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
    starts = [[0, 0], [1, 1], [-1, -1], [-1, -2], [1.95, -2], [-3, 4], [10, 10]]
    method_names_column = []
    start_column = []
    result_column = []
    result_value_column = []
    iterations_column = []
    for start in starts:
        result_directions, iters = conjugate_direction_method(Q=quadratic_function_Q(),
                                                              b=np.array([-2, 0], dtype="float64"),
                                                              start=np.array(start, dtype="float64")
                                                              )
        method_names_column.append("conjugate_dirs")
        start_column.append(start)
        result_column.append(result_directions)
        iterations_column.append(iters)
        result_value_column.append(quadratic_function(result_directions))

        result_conj_grad, iters = conjugate_gradient_method(f=quadratic_function,
                                                            f_grad=quadratic_function_grad,
                                                            start=np.array(start, dtype="float64")
                                                            )
        method_names_column.append("conjugate_grad")
        start_column.append(start)
        result_column.append(result_conj_grad)
        iterations_column.append(iters)
        result_value_column.append(quadratic_function(result_conj_grad))

        result_newton, iters = newton_method(f=quadratic_function,
                                             f_grad=quadratic_function_grad,
                                             f_hess=quadratic_function_hess,
                                             start=np.array(start, dtype="float64"))
        method_names_column.append("newton")
        start_column.append(start)
        result_column.append(result_newton)
        iterations_column.append(iters)
        result_value_column.append(quadratic_function(result_newton))

        print(result_directions)
        print(result_conj_grad)
        print(result_newton)

    table = \
        {
            'method': method_names_column,
            'start': start_column,
            'result x': result_column,
            'result f(x)': result_value_column,
            'iterations': iterations_column
        }

    return pd.DataFrame(data=table)


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

    starts = [[0, 0], [10, 10], [-1, -2], [-3, 3], [-12, -11], [2, 2]]
    method_names_column = []
    start_column = []
    result_column = []
    iterations_column = []
    result_value_column = []
    for start in starts:
        result_conj_grad, iters = conjugate_gradient_method(f=rosenbrock_function,
                                                            f_grad=rosenbrock_function_grad,
                                                            start=np.array(start, dtype="float64")
                                                            )
        method_names_column.append("conjugate_grad")
        start_column.append(start)
        result_column.append(result_conj_grad)
        iterations_column.append(iters)
        result_value_column.append(rosenbrock_function(result_conj_grad))

        result_newton, iters = newton_method(f=rosenbrock_function,
                                             f_grad=rosenbrock_function_grad,
                                             f_hess=rosenbrock_function_hess,
                                             start=np.array(start, dtype="float64"))

        method_names_column.append("newton")
        start_column.append(start)
        result_column.append(result_newton)
        iterations_column.append(iters)
        result_value_column.append(rosenbrock_function(result_newton))

        print(result_conj_grad)
        print(result_newton)

    table = \
        {
            'method': method_names_column,
            'start': start_column,
            'result x': result_column,
            'result f(x)': result_value_column,
            'iterations': iterations_column
        }

    return pd.DataFrame(data=table)


#########################################

def test_function(x: np.ndarray):
    return -2 * math.exp(-(((x[0] - 1) / 2) ** 2) - (x[1] - 1) ** 2) - 3 * math.exp(
        -(((x[0] - 2) / 3) ** 2) - ((x[1] - 3) / 2) ** 2)


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

    starts = [[3, 3], [0, 0], [-1, -1], [1.5, 1.35], [3.5, 3.5], [1.2, 1.2], [1.5, 1.5]]
    method_names_column = []
    start_column = []
    result_column = []
    iterations_column = []
    result_value_column = []
    for start in starts:
        result_conj_grad, iters = conjugate_gradient_method(f=test_function,
                                                            f_grad=test_function_grad,
                                                            start=np.array(start, dtype="float64")
                                                            )
        method_names_column.append("conjugate_grad")
        start_column.append(start)
        result_column.append(result_conj_grad)
        iterations_column.append(iters)
        result_value_column.append(test_function(result_conj_grad))

        result_newton, iters = newton_method(f=test_function,
                                             f_grad=test_function_grad,
                                             f_hess=test_function_hess,
                                             start=np.array(start, dtype="float64"))

        method_names_column.append("newton")
        start_column.append(start)
        result_column.append(result_newton)
        iterations_column.append(iters)
        result_value_column.append(test_function(result_newton))

        print(result_conj_grad)
        print(result_newton)

    table = \
        {
            'method': method_names_column,
            'start': start_column,
            'result x': result_column,
            'result f(x)': result_value_column,
            'iterations': iterations_column
        }

    return pd.DataFrame(data=table)


#########################################

COMPARISON_DIR = '../../comparison_results'


def compare():
    quadratic_frame = compare_quadratic()
    rosenbrock_frame = compare_rosenbrock_function()
    test_foo_frame = compare_test_function()

    quadratic_frame.to_csv(f'{COMPARISON_DIR}/quadratic.csv', index=False)
    rosenbrock_frame.to_csv(f'{COMPARISON_DIR}/rosenbrock.csv', index=False)
    test_foo_frame.to_csv(f'{COMPARISON_DIR}/test_foo.csv', index=False)


if __name__ == '__main__':
    compare()
