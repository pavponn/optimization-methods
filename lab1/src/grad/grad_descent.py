from typing import Callable, List, Optional
import numpy as np

import lab1.src.grad.grad_step_strategy as st
import lab1.src.grad.stop_criteria as sc


DEFAULT_EPSILON = 1e-9
DEFAULT_MAX_ITERATIONS = 1e5


def gradient_descent(f: Callable[[np.ndarray], float],
                     f_grad: Callable[[np.ndarray], np.ndarray],
                     start: np.ndarray,
                     step_strategy: st.StepStrategy,
                     stop_criteria: sc.StopCriteria,
                     eps_strategy: float = DEFAULT_EPSILON,
                     eps_stop_criteria: float = DEFAULT_EPSILON,
                     max_iterations_strategy=DEFAULT_MAX_ITERATIONS,
                     max_iterations_criteria=DEFAULT_MAX_ITERATIONS,
                     trajectory: Optional[List] = None):
    strategy = st.get_step_strategy(step_strategy, f, f_grad, eps_strategy, max_iterations_strategy)
    criteria = sc.get_stop_criteria(stop_criteria, f, f_grad, eps_stop_criteria, max_iterations_criteria)
    cur_x = start
    iters = 0

    if trajectory is not None:
        trajectory.append(cur_x)

    while True:
        iters += 1
        cur_grad = f_grad(cur_x)
        step = strategy.next_step(cur_x)
        next_x = cur_x - step * cur_grad

        if criteria.should_stop(cur_x, next_x):
            return cur_x, iters

        cur_x = next_x
        if trajectory is not None:
            trajectory.append(cur_x)

        if iters == max_iterations_criteria:
            return cur_x, iters


if __name__ == '__main__':
    def foo(p):
        return p[0] ** 2 + p[1] ** 2

    def foo_grad(p):
        x, y = p[0], p[1]
        return np.array([2 * x, 2 * y])


    res, _ = gradient_descent(foo,
                              foo_grad,
                              start=np.array([3, 4]),
                              step_strategy=st.StepStrategy.DIVIDE_STEP,
                              stop_criteria=sc.StopCriteria.BY_GRAD)
    print(res)
