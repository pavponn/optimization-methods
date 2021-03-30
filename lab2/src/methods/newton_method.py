import numpy as np
from typing import Callable
from lab2.src.methods.conjugate_method import conjugate_direction_method
from lab2.src.methods.newton_step_strategy import ConstantStepStrategy

DEFAULT_EPS = 1e-6
DEFAULT_MAX_ITERS = 100


def newton_method(f: Callable[[np.ndarray], float],
                  f_grad: Callable[[np.ndarray], np.ndarray],
                  f_hess: Callable[[np.ndarray], np.ndarray],
                  start: np.ndarray,
                  eps: float = DEFAULT_EPS,
                  max_iters: int = DEFAULT_MAX_ITERS,
                  use_conjugate: bool = False):
    x_prev = start
    iters = 0
    strategy = ConstantStepStrategy(f)

    while iters < max_iters:
        grad_prev = f_grad(x_prev)
        hess_prev = f_hess(x_prev)

        if use_conjugate:
            x_wave = conjugate_direction_method(f_hess(x_prev), f_grad(x_prev), x_prev)
            alpha = strategy.next_step(x_prev, x_wave)
            x_k = x_prev + alpha * x_wave
            if np.linalg.norm(alpha * x_wave) < eps:
                return x_k
            x_prev = x_k
        else:
            hess_inv = np.linalg.inv(hess_prev)
            val = np.matmul(grad_prev, hess_inv)
            x_k = x_prev - val
            if np.linalg.norm(val) < eps:
                return x_k
            x_prev = x_k

        iters += 1

    return x_prev
