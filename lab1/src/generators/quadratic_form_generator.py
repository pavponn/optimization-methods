import random
import numpy as np


def random_matrix(n: int):
    return np.random.rand(n, n)


# Here, we use the fact that  real matrix A is symmetric
# iff its spectral decomposition is as follows:
# A = SDS^-1, S is orthogonal (A = SDS^T)
def generate_matrix_with_condition_number(n: int, k: float):
    char_values = [random.uniform(1, k) for _ in range(n - 2)]
    char_values.extend([1.0, k])
    char_values.sort()
    d = np.diag(char_values)
    m = random_matrix(n)
    q, _ = np.linalg.qr(m)
    return np.matmul(np.matmul(q, d), np.transpose(q))


if __name__ == '__main__':
    print(generate_matrix_with_condition_number(3, 3))
