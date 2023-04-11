import numpy
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time


def calculate_2d_dft(x: np.ndarray) -> np.ndarray:
    """
    This method calculates the 2D DFT using the Summation algorithm
    :param x:
    :return: DFT Matrix with the same dimensions as x
    """
    # Get individual dimensions for iterations
    n1 = x.shape[0]
    n2 = x.shape[1]

    results = np.zeros((n1, n2), dtype=np.complex_)
    # The first two loops are for iterating over result matrix
    for i1 in range(0, n1):
        for i2 in range(0, n2):
            summation = 0
            # These two inner loops are the actual dft calculation
            for k1 in range(0, n1):
                for k2 in range(0, n2):
                    summation += x[k1][k2] * np.exp((-2 * np.pi * 1j * i1 * k1) / n1 + (-2 * np.pi * 1j * i2 * k2) / n2)
            results[i1][i2] = summation

    return results


def calculate_2d_dft_matrix(x: np.ndarray) -> np.ndarray:
    """
    This method calculates the 2D DFT using the Matrix algorithm
    :param x:
    :return: DFT Matrix with the same dimensions as x
    """
    n1 = x.shape[0]
    n2 = x.shape[1]

    n1_fourier_mtx = np.zeros((n1, n1), dtype=np.complex_)
    for i1 in range(n1):
        for i2 in range(n1):
            n1_fourier_mtx[i1][i2] = np.exp((-2 * np.pi * 1j * i1 * i2)/n1)

    n2_fourier_mtx = np.zeros((n2, n2), dtype=np.complex_)
    for i1 in range(n2):
        for i2 in range(n2):
            n2_fourier_mtx[i1][i2] = np.exp((-2 * np.pi * 1j * i1 * i2)/n2)

    n2_fourier_mtx_transpose = np.transpose(n2_fourier_mtx)
    temp_mtx = np.matmul(n1_fourier_mtx, x)
    results = np.matmul(temp_mtx, n2_fourier_mtx_transpose)
    return results


def confirm_transforms_work():
    """
    Calculate the DFT matrix using the Summation, FFT, and Matrix algorithms.
    Then print them to the console for verification they match.
    """
    print('Enter number of rows')
    rows = int(input())

    print('Enter number of columns')
    columns = int(input())

    signal = np.random.rand(rows, columns)

    two_dim_dft = calculate_2d_dft(signal)
    two_dim_dft_fft = scipy.fft.fft2(signal)
    two_dim_dft_mtx = calculate_2d_dft_matrix(signal)

    print('2D Summation')
    print(two_dim_dft)
    print('2D FFT')
    print(two_dim_dft_fft)
    print('2D MTX')
    print(two_dim_dft_mtx)


def run_time_tests():
    """
    Run incremental timed tests to test performance of each algorithm
    """
    results = []
    x = []
    for columnAndRowCount in range(10, 100, 10):
        signal = np.random.rand(columnAndRowCount, columnAndRowCount)
        calculate_2d_dft_matrix(signal)
        dft_start_time = time.time()
        two_dim_dft = calculate_2d_dft(signal)
        dft_end_time = time.time()

        fft_start_time = time.time()
        two_dim_dft_fft = scipy.fft.fft2(signal)
        fft_end_time = time.time()

        mtx_start_time = time.time()
        two_dim_dft_mtx = calculate_2d_dft_matrix(signal)
        mtx_end_time = time.time()

        results.append((dft_end_time - dft_start_time, fft_end_time - fft_start_time, mtx_end_time - mtx_start_time))
        x.append(columnAndRowCount)

    dft_results = [r[0] for r in results]
    fft_results = [r[1] for r in results]
    mtx_results = [r[2] for r in results]

    fig, ax1 = plt.subplots(1)
    ax1.plot(x, dft_results, label='2D DFT Summation')
    ax1.plot(x, mtx_results, label='2D MTX')
    ax1.plot(x, fft_results, label='2D FFT')
    ax1.set_xlabel('N Samples')
    ax1.set_ylabel('Time in Seconds')
    ax1.legend()

    plt.show()

    # Plot the two efficient algorithms only for more details on their relative performances
    fig, ax2 = plt.subplots(1)
    ax2.plot(x, mtx_results, label='2D MTX')
    ax2.plot(x, fft_results, label='2D FFT')
    ax2.set_xlabel('N Samples')
    ax2.set_ylabel('Time in Seconds')
    ax2.legend()

    plt.show()


# Skip the double summation implementation due to performance issues
def run_high_sample_trial_tests():
    """
    Run larger trials on the more efficient algorithms
    """
    results = []
    x = []
    for columnAndRowCount in range(250, 2500, 250):
        signal = np.random.rand(columnAndRowCount, columnAndRowCount)

        fft_start_time = time.time()
        two_dim_dft_fft = scipy.fft.fft2(signal)
        fft_end_time = time.time()

        mtx_start_time = time.time()
        two_dim_dft_mtx = calculate_2d_dft_matrix(signal)
        mtx_end_time = time.time()

        results.append((fft_end_time - fft_start_time, mtx_end_time - mtx_start_time))
        x.append(columnAndRowCount)

    fft_results = [r[0] for r in results]
    mtx_results = [r[1] for r in results]

    fig, ax1 = plt.subplots(1)
    ax1.plot(x, mtx_results, label='2D MTX')
    ax1.plot(x, fft_results, label='2D FFT')
    ax1.set_xlabel('N Samples')
    ax1.set_ylabel('Time in Seconds')
    ax1.legend()

    plt.show()


if __name__ == '__main__':
    confirm_transforms_work()
    run_time_tests()
    run_high_sample_trial_tests()
