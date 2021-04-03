from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

from lab1.src.grad.grad_step_strategy import StepStrategy
from lab1.src.grad.grad_descent import gradient_descent
from lab1.src.grad.stop_criteria import StopCriteria

from lab2.src.methods.newton_method import newton_method
from lab2.src.methods.conjugate_method import conjugate_direction_method, conjugate_gradient_method
from lab2.src.comparison.compare_methods import (
    quadratic_function,
    quadratic_function_grad,
    quadratic_function_hess
)


def simple_quadratic_function(x: np.ndarray):
    return x[0] ** 2 - x[0] * x[1] + 2 * x[1] ** 2


def simple_quadratic_function_grad(x: np.ndarray):
    return np.array(
        [
            2 * x[0] - x[1],
            - x[0] + 4 * x[1]
        ], dtype="float64"
    )


def simple_quadratic_function_hess(_x: np.ndarray):
    return np.array(
        [
            [2, -1],
            [-1, 4]
        ], dtype="float64"
    )


FUNCTIONS = [
    (
        "100(y - x)² + (1 - x)²",  # f_str
        quadratic_function,  # f
        quadratic_function_grad,  # f_grad
        quadratic_function_hess,  # f_hess
        np.array([-2, 0], dtype="float64"),  # b
        (0.5, -1.5),  # start
        -1, 1.5,  # x_b, x_b
        -2, 1.5,  # y_a, y_b
    ),
    (
        "x² - xy + 2y²",  # f_str
        simple_quadratic_function,  # f
        simple_quadratic_function_grad,  # f_grad
        simple_quadratic_function_hess,  # f_hess
        np.array([0, 0], dtype="float64"),  # b
        (0.5, -0.5),  # start
        -1, 1,  # x_b, x_b
        -1, 1,  # y_a, y_b
    )
]


def wrap_array_f(f: Callable[[np.array], float]):
    def result(x, y):
        return f(np.array([x, y]))
    return result


def main():
    eps = 1e-6
    delta = 0.025

    for f_str, f, f_grad, f_hess, b, start, x_a, x_b, y_a, y_b in FUNCTIONS:
        x = np.arange(x_a, x_b, delta)
        y = np.arange(y_a, y_b, delta)
        x, y = np.meshgrid(x, y)
        z = wrap_array_f(f)(x, y)

        fig, ax = plt.subplots()
        cs = ax.contour(x, y, z)
        ax.clabel(cs, inline=True, fontsize=10)
        ax.set_title(f'Function: {f_str}, start: {start}')

        trajectory = []
        gradient_descent(f, f_grad, np.array(start),
                         step_strategy=StepStrategy.FIBONACCI_STEP,
                         stop_criteria=StopCriteria.BY_ARGUMENT,
                         trajectory=trajectory,
                         eps_strategy=eps,
                         eps_stop_criteria=eps)
        plt.plot(*zip(*trajectory), c='lime', marker='o',
                 label=f'Gradient descent ({len(trajectory) - 1})')

        trajectory = []
        iters = newton_method(f, f_grad, f_hess, np.array(start),
                              trajectory=trajectory,
                              eps=eps)[1]
        plt.plot(*zip(*trajectory), c='dodgerblue',
                 label=f'Newton method ({iters})')

        trajectory = []
        iters = conjugate_direction_method(f_hess(np.array([])), b,
                                           np.array(start),
                                           trajectory=trajectory,
                                           eps=eps)[1]
        plt.plot(*zip(*trajectory), c='r',
                 label=f'Conjugate direction ({iters})')

        trajectory = []
        iters = conjugate_gradient_method(f, f_grad,
                                          np.array(start),
                                          trajectory=trajectory,
                                          eps=eps)[1]
        plt.plot(*zip(*trajectory), c='purple',
                 label=f'Conjugate gradients ({iters})')

        plt.plot(*start, 'mx')
        plt.legend()
        plt.savefig(f"../../comparison_results/trajectory/{f_str}_{start}.png")


if __name__ == '__main__':
    main()
