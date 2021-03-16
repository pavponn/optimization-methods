import logging
import inspect


def log_init_one_dim_method(method_name, foo, a, b, eps):
    lines = inspect.getsource(foo)
    logging.info(f'method "{method_name}" launched, foo = {lines.strip()}, on segment from {a} to {b} with eps={eps}')


def log_cur_segment(a, b):
    logging.info(f'current segment: [{a}, {b}]')


def log_method_finished():
    logging.info(f'Method finished')
    logging.info(f'===============================')
