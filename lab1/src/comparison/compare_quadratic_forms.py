import itertools
import pandas as pd
from tqdm import tqdm

from lab1.src.grad.grad_descent import (
    gradient_descent
)

from lab1.src.generators.optimization_task_generator import (
    generate_optimization_task
)

import lab1.src.grad.stop_criteria as sc
import lab1.src.grad.grad_step_strategy as ss

COMPARISON_DIR = '../../comparison_results/quadratic'

STRATEGIES = [ss.StepStrategy.FIBONACCI_STEP, ss.StepStrategy.DIVIDE_STEP, ss.StepStrategy.CONSTANT_STEP]

ns_test = [2, 3, 4, 5, 6, 7, 8, 9, 10]
ks_test = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 15, 20, 25, 30, 45, 50, 70, 100]

n_and_k = list(itertools.product(ns_test, ks_test))


def compare(strategy: ss.StepStrategy):
    ns = []
    ks = []
    iterations = []
    for (n, k) in tqdm(n_and_k):
        f, f_grad = generate_optimization_task(n, k)
        default_start = n * [100]
        _, iters = gradient_descent(f, f_grad, default_start, strategy, sc.StopCriteria.BY_FUNCTION)
        ns.append(n)
        ks.append(k)
        iterations.append(iters)
        # print(f'n={n},k={k},iters={iters}')

    data = {
        'n': ns,
        'k': ks,
        'iters': iterations
    }

    return pd.DataFrame(data=data)


def compare_and_write_to_files():
    for strategy in STRATEGIES:
        frame = compare(strategy)
        frame.to_csv(f'{COMPARISON_DIR}/{strategy}_quadratic.csv', index=False)


if __name__ == '__main__':
    compare_and_write_to_files()
