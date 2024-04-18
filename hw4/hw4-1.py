import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Helvetica'


def solve_embedding(D, d):
    """
    Solve the low-rank embedding of the symmetric matrix with MDS
    """
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    G = - 0.5 * H @ (D * D) @ H
    eigenvalues, eigenvectors = np.linalg.eig(G)
    idx = np.argsort(eigenvalues)[::-1][:d]
    sigma = np.diag(eigenvalues[idx])
    Q = eigenvectors[:, idx]
    return Q.dot(np.sqrt(sigma))


def euclidean_distance(vec1, vec2):
    """
    Calculate the Euclidean distance between two vectors
    """
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def distance(M):
    """
    Calculate the sqaured distance between row vectors
    """
    n = M.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = D[j, i] = euclidean_distance(M[i, :], M[j, :])
    return D


if __name__ == '__main__':
    # Load the data
    data = pd.read_csv("../data/hw4/city_distance_data.csv", index_col=0)
    dist = data.values

    # Solve the embedding
    embedding = solve_embedding(dist, 2)

    # Calculate the distance between the embedded points
    est_dist = distance(embedding)

    # Plot the embedded points
    plt.plot(embedding[:, 0], embedding[:, 1], 'o')
    for i in range(embedding.shape[0]):
        plt.text(embedding[i, 0], embedding[i, 1]+30, data.index[i], fontsize=10, ha='center')
    plt.title("2D MDS of City Distance Data")
    plt.savefig("./output/hw4-1.png")
    plt.show()