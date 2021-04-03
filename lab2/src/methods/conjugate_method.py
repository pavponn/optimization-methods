from typing import Callable, List, Optional
from lab1.src.onedim.one_dim_search import dichotomy_method

import numpy as np


DEFAULT_EPS = 1e-6
DEFAULT_MAX_ITERS = 100


def conjugate_direction_method(q: np.ndarray,
                               b: np.ndarray,
                               start: np.ndarray,
                               max_iters=DEFAULT_MAX_ITERS,
                               eps: float = DEFAULT_EPS,
                               trajectory: Optional[List] = None):
    if np.all(b == 0):
        return b, 0

    if trajectory is not None:
        trajectory.append(start)
    w_prev = - q @ start - b
    u_prev = w_prev
    if np.linalg.norm(u_prev) == 0:
        return start, 0
    h_prev = np.dot(u_prev, u_prev) / np.dot(q @ u_prev, u_prev)
    x_prev = start + h_prev * u_prev
    if trajectory is not None:
        trajectory.append(x_prev)

    k = 1
    while k < max_iters:
        w_k = -q @ x_prev - b
        u_k = w_k - (np.dot(q @ u_prev, w_k) / np.dot(q @ u_prev, u_prev)) * u_prev
        if np.linalg.norm(u_k) < eps:
            return x_prev, k
        h_k = np.dot(w_k, u_k) / np.dot(q @ u_k, u_k)
        x_k = x_prev + h_k * u_k

        u_prev = u_k
        x_prev = x_k
        k += 1
        if trajectory is not None:
            trajectory.append(x_prev)

    return x_prev, k


def conjugate_gradient_method(f: Callable[[np.ndarray], float],
                              f_grad: Callable[[np.ndarray], np.ndarray],
                              start: np.ndarray,
                              eps=DEFAULT_EPS,
                              max_iters=DEFAULT_MAX_ITERS,
                              trajectory: Optional[List] = None):
    if trajectory is not None:
        trajectory.append(start)

    w_prev = (-1) * f_grad(start)
    u_prev = w_prev
    if np.linalg.norm(w_prev) < eps:
        return start, 0
    x_prev = start
    k = 1

    while k < max_iters:
        w_k = (-1) * f_grad(x_prev)
        y_k = max(0, np.dot(w_k - w_prev, w_k) / np.dot(w_prev, w_prev))
        u_k = w_k + y_k * u_prev
        h_k, _, _ = dichotomy_method(lambda h: f(x_prev + h * u_k), 0, 50, 1e-8)
        x_k = x_prev + h_k * u_k

        if np.linalg.norm(w_k) < eps or np.linalg.norm(x_k - x_prev) < eps:
            return x_prev, k

        x_prev = x_k
        u_prev = u_k
        w_prev = w_k
        k += 1

        if trajectory is not None:
            trajectory.append(x_prev)

    return x_prev, k
