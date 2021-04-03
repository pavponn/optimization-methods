import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

from lab1.src.grad.grad_step_strategy import StepStrategy
from lab1.src.grad.grad_descent import gradient_descent
from lab1.src.grad.stop_criteria import StopCriteria

from lab2.src.methods.newton_method import newton_method
from lab2.src.methods.conjugate_method import (
    conjugate_direction_method,
    conjugate_gradient_method
)


EPS = 1e-7

FUNCTIONS = [
    (
        "100(y - x)² + (1 - x)² + (z - y)² + 5(x + y)²",  # f_str
        lambda x: 100 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2 + (x[2] - x[1]) ** 2 + 5 * (x[0] + x[1]) ** 2,  # f
        lambda x: np.array([
            212 * x[0] - 190 * x[1] - 2,
            -190 * x[0] + 212 * x[1] - 2 * x[2],
            -2 * x[1] + 2 * x[2]
        ], dtype="float64"),  # f_grad
        lambda x: np.array([
                [212, -190, 0],
                [-190, 212, -2],
                [0, -2, 2]
            ], dtype="float64"),  # f_hess
        np.array([-2, 0, 0], dtype="float64"),  # b
        (1e8, -1e8, 2 * 1e7),  # start
    )
]


def run_grad_descent(f, f_grad, _f_hess, _b, start):
    trajectory = []
    gradient_descent(f, f_grad, np.array(start),
                     step_strategy=StepStrategy.FIBONACCI_STEP,
                     stop_criteria=StopCriteria.BY_ARGUMENT,
                     trajectory=trajectory,
                     eps_strategy=EPS,
                     eps_stop_criteria=EPS)
    iters = len(trajectory) - 1
    return iters


def run_newton_method(f, f_grad, f_hess, _b, start):
    iters = newton_method(f, f_grad, f_hess, np.array(start), eps=EPS)[1]
    return iters


def run_conj_dir_method(_f, _f_grad, f_hess, b, start):
    iters = conjugate_direction_method(
        f_hess(np.array([])), b,
        np.array(start), eps=EPS
    )[1]
    return iters


def run_conj_grad_method(f, f_grad, _f_hess, _b, start):
    iters = conjugate_gradient_method(f, f_grad, np.array(start), eps=EPS)[1]
    return iters


METHODS = {
    'Gradient descent': run_grad_descent,
    'Newton method': run_newton_method,
    'Conjugate direction': run_conj_dir_method,
    'Conjugate gradient': run_conj_grad_method,
}


def plot_memory_usage(method_name: str, mem_usage: [float]):
    print(f"Peak memory usage: {max(mem_usage):.2f} MiB")
    plt.plot([i * 0.05 for i in range(len(mem_usage))], mem_usage)
    plt.title(f"Memory usage by {method_name} (MiB)")
    plt.show()


def main():
    for f_str, f, f_grad, f_hess, b, start in FUNCTIONS:
        for (method_name, method) in METHODS.items():
            print(f"# Running {method_name}...")
            mem_usage, iters = memory_usage(
                (method, (f, f_grad, f_hess, b, start)),
                interval=0.05,
                retval=True
            )
            print(f"Finished in {iters} iterations")
            plot_memory_usage(method_name, mem_usage)


if __name__ == '__main__':
    main()
