import math

import lab1.src.utils.logger as lg


DEFAULT_EPSILON = 1e-7
DEFAULT_MAX_ITERATIONS = 100

GOLDEN_RATION_CONSTANT = (1 + math.sqrt(5)) / 2


def dichotomy_method(f, a, b, eps=DEFAULT_EPSILON, max_iter=DEFAULT_MAX_ITERATIONS, enable_logging=False):
    """
    :param f: function to find minimum for (should be unimodal on [a, b])
    :param a: interval start
    :param b: interval end
    :param eps: method precision
    :param max_iter: maximum number of method iterations
    :param enable_logging: if set to True enables logging according to basic logging specification
    :return: the minimum point on [a,b], number of iterations, number of function f computations
    """
    if enable_logging:
        lg.log_init_one_dim_method("dichotomy method", f, a, b, eps)

    a_k, b_k = a, b
    delta = eps / 2 - 1e-10
    iters = 0

    while abs(a_k - b_k) > 2 * eps and iters < max_iter:
        if enable_logging:
            lg.log_cur_segment(a_k, b_k)

        iters += 1

        x_1 = (a_k + b_k) / 2 - delta
        x_2 = (a_k + b_k) / 2 + delta
        if f(x_1) >= f(x_2):
            a_k = x_1
        else:
            b_k = x_2

    if enable_logging:
        lg.log_method_finished()

    return (a_k + b_k) / 2, iters, iters * 2


def golden_section_method(f, a, b, eps=DEFAULT_EPSILON, max_iter=DEFAULT_MAX_ITERATIONS, enable_logging=False):
    """
    :param f: function to find minimum for (should be unimodal on [a, b])
    :param a: interval start
    :param b: interval end
    :param eps: method precision
    :param max_iter: maximum number of method iterations
    :param enable_logging: if set to True enables logging according to basic logging specification
    :return: the minimum point on [a,b], number of iterations, number of function f computations
    """
    if enable_logging:
        lg.log_init_one_dim_method("golden section method", f, a, b, eps)

    a_k, b_k = a, b
    x_1 = b_k - abs(b_k - a_k) / GOLDEN_RATION_CONSTANT
    x_2 = a_k + abs(b_k - a_k) / GOLDEN_RATION_CONSTANT
    f_prev = f(x_1)
    use_x_1 = True
    iters = 0

    while abs(b_k - a_k) >= eps and iters < max_iter:
        if enable_logging:
            lg.log_cur_segment(a_k, b_k)

        iters += 1
        if use_x_1:
            f_1 = f_prev
            f_2 = f(x_2)
        else:
            f_1 = f(x_1)
            f_2 = f_prev

        if f_1 >= f_2:
            a_k = x_1
            x_1 = x_2
            x_2 = a_k + abs(b_k - a_k) / GOLDEN_RATION_CONSTANT
            f_prev = f_2
            use_x_1 = True
        else:
            b_k = x_2
            x_2 = x_1
            x_1 = b_k - abs(b_k - a_k) / GOLDEN_RATION_CONSTANT
            f_prev = f_1
            use_x_1 = False

    if enable_logging:
        lg.log_method_finished()

    return (b_k + a_k) / 2, iters, iters + 1


def fibonacci_method(f, a, b, eps=DEFAULT_EPSILON, enable_logging=False):
    """
    :param f: function to find minimum for (should be unimodal on [a, b])
    :param a: interval start
    :param b: interval end
    :param eps: method precision
    :param enable_logging: if set to True enables logging according to basic logging specification
    :return: the minimum point on [a,b], number of iterations, number of function f computations
    """
    if enable_logging:
        lg.log_init_one_dim_method("fibonacci method", f, a, b, eps)

    a_k, b_k = a, b
    n, fibs = get_n_and_fibs(a, b, eps)
    n = n - 1  # for indexing
    k = 1
    x_1 = a_k + fibs[n] / fibs[n + 2] * (b_k - a_k)
    x_2 = a_k + fibs[n + 1] / fibs[n + 2] * (b_k - a_k)
    f_prev = f(x_1)
    use_x_1 = True

    while k < n:
        if enable_logging:
            lg.log_cur_segment(a_k, b_k)

        if use_x_1:
            f_1 = f_prev
            f_2 = f(x_2)
        else:
            f_1 = f(x_1)
            f_2 = f_prev

        if f_1 >= f_2:
            a_k = x_1
            x_1 = x_2
            f_prev = f_2
            x_2 = a_k + fibs[n - k + 2] / fibs[n - k + 3] * (b_k - a_k)
            use_x_1 = True
        else:
            b_k = x_2
            x_2 = x_1
            f_prev = f_1
            x_1 = a_k + fibs[n - k + 1] / fibs[n - k + 3] * (b_k - a_k)
            use_x_1 = False
        k += 1

    if enable_logging:
        lg.log_method_finished()

    return (x_1 + x_2) / 2, k, k + 1


def get_n_and_fibs(a, b, eps):
    fibs = generate_fib(a, b, eps)
    n = len(fibs) - 2
    return n, fibs


def generate_fib(a, b, eps):
    val = (b - a) / eps
    fibs = [1, 1]
    k = 2
    while True:
        fibs.append(fibs[k - 1] + fibs[k - 2])
        if fibs[k] > val:
            break
        k += 1
    return fibs
