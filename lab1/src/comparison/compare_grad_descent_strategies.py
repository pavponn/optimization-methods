import pandas as pd
import numpy as np
from tqdm import tqdm

from lab1.src.grad.grad_descent import (
    gradient_descent
)

import lab1.src.grad.stop_criteria as sc
import lab1.src.grad.grad_step_strategy as ss

COMPARISON_DIR = '../../comparison_results/grad'

STOP_CRITERIA = [
    sc.StopCriteria.BY_FUNCTION,
    sc.StopCriteria.BY_GRAD,
    sc.StopCriteria.BY_ARGUMENT
]

CASES_FOR_COMPARISON = [
    (
        lambda x: (x[0] - 5) ** 2 + 1,
        lambda x: np.array([2 * (x[0] - 5)]),
        [44]
    ),
    (
        lambda x: x[0] ** 2 + x[1] ** 2,
        lambda x: np.array([2 * x[0], 2 * x[1]]),
        [411, -322]
    ),
    (
        lambda x: 7 * (x[0] - 1) ** 2 + x[1] ** 2,
        lambda x: np.array([14 * (x[0] - 1), 2 * x[1]]),
        [400, -300]
    ),
    (
        lambda x: (x[0] - 2) ** 2 + (x[1] + 3) ** 2,
        lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] + 3)]),
        [61, 7]
    ),
    (
        lambda x: -1 / (2 * x[0] ** 2 + x[1] ** 2 + 7),
        lambda x: np.array([
            4 * x[0] / (2 * x[0] ** 2 + x[1] ** 2 + 7) ** 2,
            2 * x[1] / (2 * x[0] ** 2 + x[1] ** 2 + 7) ** 2
        ]),
        [-0.1, 0.1]
    )
]

MAX_ITERATIONS_CRITERIA = 1e4
MAX_ITERATIONS_STRATEGY = 1000

EPS_CRITERIA = 1e-6
EPS_STRATEGY = 1e-7


def compare(stop_criteria: sc.StopCriteria):
    dichotomy_iterations = []
    golden_selection_iterations = []
    fibonacci_iterations = []
    for i, (f, f_grad, start) in tqdm(enumerate(CASES_FOR_COMPARISON)):
        _, dich_iters = gradient_descent(f=f,
                                         f_grad=f_grad,
                                         start=start,
                                         step_strategy=ss.StepStrategy.DICHOTOMY_STEP,
                                         stop_criteria=stop_criteria,
                                         eps_strategy=EPS_STRATEGY,
                                         eps_stop_criteria=EPS_CRITERIA,
                                         max_iterations_strategy=MAX_ITERATIONS_STRATEGY,
                                         max_iterations_criteria=MAX_ITERATIONS_CRITERIA,
                                         )

        _, golden_iters = gradient_descent(f=f,
                                           f_grad=f_grad,
                                           start=start,
                                           step_strategy=ss.StepStrategy.GOLDEN_SELECTION_STEP,
                                           stop_criteria=stop_criteria,
                                           eps_strategy=EPS_STRATEGY,
                                           eps_stop_criteria=EPS_CRITERIA,
                                           max_iterations_strategy=MAX_ITERATIONS_STRATEGY,
                                           max_iterations_criteria=MAX_ITERATIONS_CRITERIA,
                                           )
        _, fib_iters = gradient_descent(f=f,
                                        f_grad=f_grad,
                                        start=start,
                                        step_strategy=ss.StepStrategy.FIBONACCI_STEP,
                                        stop_criteria=stop_criteria,
                                        eps_strategy=EPS_STRATEGY,
                                        eps_stop_criteria=EPS_CRITERIA,
                                        max_iterations_strategy=MAX_ITERATIONS_STRATEGY,
                                        max_iterations_criteria=MAX_ITERATIONS_CRITERIA,
                                        )

        dichotomy_iterations.append(dich_iters)
        golden_selection_iterations.append(golden_iters)
        fibonacci_iterations.append(fib_iters)

    d_iters = \
        {

            'dich_iters': dichotomy_iterations,
            'gold_iters': golden_selection_iterations,
            'fib_iters': fibonacci_iterations
        }

    return pd.DataFrame(data=d_iters)


def compare_and_write_to_files():
    for stop_criteria in tqdm(STOP_CRITERIA):
        frame = compare(stop_criteria)
        frame.to_csv(f'{COMPARISON_DIR}/{stop_criteria}_iters.csv', index=False)


if __name__ == '__main__':
    compare_and_write_to_files()
