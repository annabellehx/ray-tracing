import numpy as np
import matplotlib.pyplot as plt

def plot_matrix(matrix, N):
    assert N * N == matrix.shape[0]
    plt.xlim(0,N)
    plt.ylim(0,N)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.imshow(matrix.reshape((N,N)), cmap = 'gray')
    # plt.colorbar(fraction = 0.045, pad = 0.05)
    plt.show()

matrix = np.loadtxt("matrix.txt")
plot_matrix(matrix, 1000)
