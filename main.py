import numpy
import numpy as np
import scipy
import matplotlib.pyplot as plt


def calculate_2d_dft(x: np.ndarray) -> np.ndarray:
    # Get individual dimensions for iterations
    n1 = x.shape(0)
    n2 = x.shape(1)

    results = np.ndarray((n1, n2), dtype=np.complex_)
    # The first two loops are for iterating over result matrix
    for i1 in range(0, n1):
        for i2 in range(0, n2):
            summation = 0
            # These two inner loops are the actual dft calculation
            for k1 in range(0, n1):
                for k2 in range(0, n2):
                    first_exp_term = np.divide(np.multiply(-2, np.pi, 1j, i1, k1), n1)
                    second_exp_term = np.divide(np.multiply(-2, np.pi, 1j, i2, k2), n2)
                    summation += np.multiply(x[k1][k2], np.exp(first_exp_term + second_exp_term))
        results[i1][i2] = summation


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
