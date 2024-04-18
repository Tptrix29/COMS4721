import heapq
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'Helvetica'


def read_data(file_path):
    """
    Load the data from the file
    """
    data = []
    f = open(file_path, 'r')
    for line in f:
        data.append(line.strip().split())
    return np.array(data, dtype=float)


def PCA(X, d=2):
    """
    Perform PCA on the data
    """
    norm_X = X - np.mean(X, axis=0)
    cov = np.cov(norm_X.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, idx]
    return norm_X @ sorted_eigenvectors[:, :d]


def euclidean_distance(vec1, vec2):
    """
    Calculate the Euclidean distance between two vectors
    """
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def knn_graph(X, k):
    """
    Construct the k-nearest neighbor graph
    """
    n = X.shape[0]
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            G[i, j] = G[j, i] = euclidean_distance(X[i, :], X[j, :])
        mask = np.argsort(G[i, :])[k+1:]
        G[i, mask] = np.inf
    return G


def dijkstra(G, s):
    """
    Dijkstra algorithm to calculate the shortest path distance
    """
    n = G.shape[0]
    dist = np.full(n, np.inf)
    dist[s] = 0
    visited = np.zeros(n, dtype=bool)
    heap = [(0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v in range(n):
            if not visited[v] and G[u, v] != np.inf:
                if dist[v] > dist[u] + G[u, v]:
                    dist[v] = dist[u] + G[u, v]
                    heapq.heappush(heap, (dist[v], v))
    return dist


def path_distance(G):
    """
    Calculate the shortest path distance between all pairs of nodes
    """
    n = G.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        print(i+1)
        D[i, :] = dijkstra(G, i)
    return D


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


class EmbeddingOptimizer:
    """
    Non-linear embedding optimizer with gradient descent
    """
    def __init__(self, D, d=2, lr=1e-4):
        self.D = D  # path distance matrix
        self.lr = lr  # learning rate
        self.n = D.shape[0]  # dataset size
        self.Y = np.random.normal(0, 1, (self.n, d))  # embedding matrix
        self.gradient = np.zeros((self.n, d))  # gradient of the embedding matrix
        self.embedding_dist = distance(self.Y)  # distance between the embedded points
        self.loss_value = []  # loss value record

    def run(self, iteration=50):
        """
        Run the optimization process
        """
        while iteration:
            self.step()
            self.embedding_dist = distance(self.Y)
            self.loss_value.append(self.loss())
            iteration -= 1
            print(f"Current Loss: {self.loss()}, Remaining Iteration: {iteration}")

    def step(self):
        """
        Update the embedding matrix with gradient descent algorithm
        """
        K = (1 - self.D / (self.embedding_dist + np.eye(self.n)))
        self.gradient = 4 * (K.sum(axis=1).reshape((-1, 1)) * self.Y - K @ self.Y)
        self.Y -= self.lr * self.gradient

    def loss(self):
        """
        Calculate the loss value
        """
        return np.sum((self.embedding_dist - self.D) ** 2)


if __name__ == '__main__':
    # TODO: customize the parameters
    ################################################
    iteration = 50
    k = 10
    run_dist = False  # set to False if the path distance matrix is already calculated
    data_name = "swiss_roll"  # "swiss_roll", "swiss_roll_hole"
    data_path = "../data/hw4/"  # data file path
    array_path = "./array-data/"  # array file path
    output_path = "./output/"  # output file path
    ################################################

    file = f"{data_path}{data_name}.txt"  # data file path
    knn_filename = f"{array_path}{data_name}_knn_{k}.npy"
    output_filename = f"{array_path}{data_name}_embedding.npy"

    # Load the data
    print(f"Loading data from {file}...")
    data = read_data(file)
    data = np.array(data)

    # 3D Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"3D Data Visualization of {" ".join(data_name.split("_")).title()}")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    plt.savefig(f"{output_path}{data_name}_3d.png")
    # plt.show()

    # PCA
    print("Running PCA...")
    reduced_data = PCA(data, 2)
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'o')
    plt.title(f"2D PCA Embedding of {" ".join(data_name.split("_")).title()} Data")
    plt.savefig(f"{output_path}{data_name}_pca.png")
    print("PCA done.")
    # plt.show()

    # KNN adjacency matrix and path distance
    if run_dist:
        print("Running KNN adjacency matrix...")
        G = knn_graph(data, k)
        print("KNN adjacency matrix done.")
        print("Running path distance...")
        D = path_distance(G)
        print("Path distance done.")
        np.save(knn_filename, D)
        print("Saved path distance matrix.")

    # Load path distance matrix
    D = np.load(knn_filename)

    # Solve the embedding with gradient descent method
    print("Running embedding optimizer...")
    optimizer = EmbeddingOptimizer(D)
    optimizer.run(iteration=iteration)
    # Save the embedding result
    np.save(output_filename, optimizer.Y)
    print("Saved embedding matrix.")

    # Visualize the loss curve
    plt.plot(np.arange(len(optimizer.loss_value))+1, optimizer.loss_value, "-")
    plt.title(f"Loss Curve of {" ".join(data_name.split("_")).title()} Data")
    # plt.show()

    # Visualize the embedding result
    plt.plot(optimizer.Y[:, 0], optimizer.Y[:, 1], 'o')
    plt.title(f"2D Embedding of {" ".join(data_name.split("_")).title()} Data")
    plt.savefig(f"{output_path}{data_name}_embedding.png")
    # plt.show()




