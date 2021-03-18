from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

from lab1.src.grad.grad_step_strategy import StepStrategy, step_to_str
from lab1.src.grad.grad_descent import gradient_descent
from lab1.src.grad.stop_criteria import StopCriteria


FUNCTIONS = [
    (
        "x²+y²",
        lambda x: x[0]**2 + x[1]**2,
        lambda x: np.array([2 * x[0], 2 * x[1]]),
        -10.0, 10.0, -10.0, 10.0
    ),
    (
        "10x²+y²",
        lambda x: x[0]**2 * 10 + x[1]**2,
        lambda x: np.array([20 * x[0], 2 * x[1]]),
        -10.0, 10.0, -10.0, 10.0
    ),
    (
        "x²-xy+2y²",
        lambda x: x[0]**2 - x[0]*x[1] + 2 * x[1]**2,
        lambda x: np.array([2*x[0] - x[1], -x[0] + 4*x[1]]),
        -1.0, 1.0, -1.0, 1.0
    ),
]


def wrap_array_f(f: Callable[[np.array], float]):
    def result(x, y):
        return f(np.array([x, y]))
    return result


def main():
    delta = 0.025
    colors = ['lightcoral', 'gold', 'yellow', 'lime', 'dodgerblue',
              'navy', 'indigo', 'teal', 'pink',
              'b', 'g', 'r', 'c', 'm', 'y']

    for f_str, f, f_grad, x_a, x_b, y_a, y_b in FUNCTIONS:
        for start in [((x_a + x_b) / 2, y_a), (x_a, y_a), (x_b, y_b)]:
            x = np.arange(x_a, x_b, delta)
            y = np.arange(y_a, y_b, delta)
            x, y = np.meshgrid(x, y)
            z = wrap_array_f(f)(x, y)

            fig, ax = plt.subplots()
            cs = ax.contour(x, y, z)
            ax.clabel(cs, inline=True, fontsize=10)
            ax.set_title(f'Function: {f_str}, start: {start}')
            color_iter = 0

            for strategy in StepStrategy:
                trajectory = []
                gradient_descent(f, f_grad, np.array(start),
                                 step_strategy=strategy,
                                 stop_criteria=StopCriteria.BY_GRAD,
                                 trajectory=trajectory)
                color = colors[color_iter]
                color_iter += 1
                plt.plot(*zip(*trajectory), c=color,
                         label=f'{step_to_str(strategy)} ({len(trajectory)})')

            plt.legend()
            plt.show()


if __name__ == '__main__':
    main()
