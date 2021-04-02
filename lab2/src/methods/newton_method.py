import numpy as np
from typing import Callable, List, Optional
from lab1.src.onedim.one_dim_search import dichotomy_method
from lab2.src.methods.conjugate_method import conjugate_direction_method
from lab2.src.methods.newton_step_strategy import ConstantStepStrategy

DEFAULT_EPS = 1e-6
DEFAULT_MAX_ITERS = 1000


def newton_method(f: Callable[[np.ndarray], float],
                  f_grad: Callable[[np.ndarray], np.ndarray],
                  f_hess: Callable[[np.ndarray], np.ndarray],
                  start: np.ndarray,
                  eps: float = DEFAULT_EPS,
                  max_iters: int = DEFAULT_MAX_ITERS,
                  trajectory: Optional[List] = None):
    x_prev = start
    if trajectory is not None:
        trajectory.append(start)
    iters = 0
    strategy = ConstantStepStrategy(f, 1e-10)

    while iters < max_iters:

        x_wave, _ = conjugate_direction_method(f_hess(x_prev), f_grad(x_prev), x_prev)
        # alpha = strategy.next_step(x_prev, x_wave)
        alpha, _, _ = dichotomy_method(lambda a: f(x_prev + a * x_wave), 0, 10, 1e-9)
        x_k = x_prev + alpha * x_wave
        if trajectory is not None:
            trajectory.append(x_k)
        if np.linalg.norm(x_prev - x_k) < eps:
            return x_k, iters
        x_prev = x_k

        iters += 1

    return x_prev, iters
