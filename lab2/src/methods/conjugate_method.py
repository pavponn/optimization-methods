from typing import Callable
from lab1.src.onedim.one_dim_search import fibonacci_method

import numpy as np

DEFAULT_EPS = 1e-5
DEFAULT_MAX_ITERS = 1000


def conjugate_direction_method(Q: np.ndarray,
                               b: np.ndarray,
                               start: np.ndarray,
                               max_iters=DEFAULT_MAX_ITERS,
                               eps: float = DEFAULT_EPS):
    if np.all(b == 0):
        return b

    w_prev = - Q @ start - b
    u_prev = w_prev
    if np.linalg.norm(u_prev) == 0:
        return start
    h_prev = np.dot(u_prev, u_prev) / np.dot(Q @ u_prev, u_prev)
    x_prev = start + h_prev * u_prev

    for k in range(1, max_iters):
        w_k = -Q @ x_prev - b
        u_k = w_k - (np.dot(Q @ u_prev, w_k) / np.dot(Q @ u_prev, u_prev)) * u_prev
        if np.linalg.norm(u_k) < eps:
            return x_prev
        h_k = np.dot(w_k, u_k) / np.dot(Q @ u_k, u_k)
        x_k = x_prev + h_k * u_k

        u_prev = u_k
        x_prev = x_k

    return x_prev


def conjugate_gradient_method(f: Callable[[np.ndarray], float],
                              f_grad: Callable[[np.ndarray], np.ndarray],
                              start: np.ndarray,
                              eps=DEFAULT_EPS,
                              max_iters=DEFAULT_MAX_ITERS):
    w_prev = (-1) * f_grad(start)
    u_prev = w_prev
    if np.linalg.norm(w_prev) < eps:
        return start
    x_prev = start

    for k in range(1, max_iters):
        h_k, _, _ = fibonacci_method(lambda h: f(x_prev + h * u_prev), 0, 100, eps)
        w_k = (-1) * f_grad(x_prev)
        y_k = max(0, np.dot(w_k - w_prev, w_k) / np.dot(w_prev, w_prev))
        u_k = w_k + y_k * u_prev
        x_k = x_prev + h_k * u_k

        if np.linalg.norm(w_k) < eps:
            return x_prev

        x_prev = x_k
        u_prev = u_k
        w_prev = w_k

    return x_prev
