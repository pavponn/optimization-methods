import math

DEFAULT_EPSILON = 1e-7
DEFAULT_MAX_ITERATIONS = 50

GOLDEN_RATION_CONSTANT = (1 + math.sqrt(5)) / 2


def dichotomy_method(f, a, b, eps=DEFAULT_EPSILON, max_iter=DEFAULT_MAX_ITERATIONS):
    a_i, b_i = a, b
    delta = eps / 2 - 1e-10
    iters = 0
    while abs(a_i - b_i) >= eps and iters < max_iter:
        # print(f'a_i: {a_i}, b_i: {b_i}, length: {abs(b_i - a_i)}')
        iters += 1
        x_1 = (a_i + b_i) / 2.0 - delta
        x_2 = (a_i + b_i) / 2.0 + delta
        # print(f'x_1: {x_1}, x_2: {x_2}')
        f_1 = f(x_1)
        f_2 = f(x_2)
        if f_1 >= f_2:
            a_i = x_1
        else:
            b_i = x_2

    return (a_i + b_i) / 2.00, iters


def golden_selection_method(f, a, b, eps=DEFAULT_EPSILON, max_iter=DEFAULT_MAX_ITERATIONS):
    a_i, b_i = a, b
    x_1 = b_i - abs(b_i - a_i) / GOLDEN_RATION_CONSTANT
    x_2 = a_i + abs(b_i - a_i) / GOLDEN_RATION_CONSTANT
    f_prev = f(x_1)
    use_x_1 = True
    iters = 0

    while abs(b_i - a_i) >= eps and iters < max_iter:
        iters += 1
        if use_x_1:
            f_2 = f(x_2)
            f_1 = f_prev
        else:
            f_1 = f(x_1)
            f_2 = f_prev

        if f_1 >= f_2:
            a_i = x_1
            x_1 = x_2
            x_2 = a_i + abs(b_i - a_i) / GOLDEN_RATION_CONSTANT
            f_prev = f_2
            use_x_1 = True
        else:
            b_i = x_2
            x_2 = x_1
            x_1 = b_i - abs(b_i - a_i) / GOLDEN_RATION_CONSTANT
            use_x_1 = False
            f_prev = f_1

    return (b_i + a_i) / 2, iters


# TODO: doesn't work properly
def fibonacci_method(f, a, b, eps=DEFAULT_EPSILON):
    a_k, b_k = a, b
    n, fibs = get_n_and_fibs(a, b, eps)
    # for index
    n = n - 1
    k = 1
    x_1 = a_k + fibs[n] / fibs[n + 2] * (b_k - a_k)
    x_2 = a_k + fibs[n + 1] / fibs[n + 2] * (b_k - a_k)
    f_prev = f(x_1)
    use_x_1 = True
    while k < n:
        # print(f'x1: {x_1}, x2: {x_2}, length: {abs(x_2 - x_1)}')
        if use_x_1:
            f_1 = f_prev
            f_2 = f(x_2)
        else:
            f_1 = f(x_1)
            f_2 = f_prev

        if f_1 >= f_2:
            a_k = x_1
            x_1 = x_2
            use_x_1 = True
            f_prev = f_2
            x_2 = a_k + fibs[n - k + 2] / fibs[n - k + 3] * (b_k - a_k)
        else:
            b_k = x_1
            x_2 = x_1
            use_x_1 = False
            f_prev = f_1
            x_1 = a_k + fibs[n - k + 1] / fibs[n - k + 3] * (b_k - a_k)
        k += 1
    return (x_1 + x_2) / 2, k


def get_n_and_fibs(a, b, eps):
    fibs = generate_fib(a, b, eps)
    n = len(fibs) - 2
    return n, fibs


def generate_fib(a, b, eps):
    val = b - a / eps
    fibs = [1, 1]
    k = 2
    while True:
        fibs.append(fibs[k - 1] + fibs[k - 2])
        if fibs[k] > val:
            break
        k += 1
    return fibs

