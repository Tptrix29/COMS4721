import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'Helvetica'


def simulate(mu, sigma, n):
    """
    Generate samples from multivariate normal distribution
    """
    return np.random.multivariate_normal(mu, sigma, n)


def exp_kernel(x1, x2, h=5):
    """
    Exponential kernel
    """
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / h)


def periodic_kernel(x1, x2, h=5, p=3, sigma=1):
    """
    Periodic kernel
    """
    return np.exp(-2 * np.sin(np.pi * np.linalg.norm(x1 - x2) / p) ** 2 / (h ** 2)) * sigma ** 2


def kernel_matrix(X, kernel, h=5):
    """
    Calculate the covariance matrix with specific kernel function
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = kernel(X[i, :], X[j, :], h)
    return K


def posterior_inference(X, X_train, Y_train, kernel):
    """
    Calculate the posterior distribution of the Gaussian process
    """
    n, m = X.shape[0], X_train.shape[0]
    _X = np.hstack([X, X_train])
    sigma = kernel_matrix(_X.reshape((m+n, -1)), kernel)
    sigma11, sigma12, sigma21, sigma22 = sigma[:n, :n], sigma[:n, n:], sigma[n:, :n], sigma[n:, n:]
    mu1 = np.zeros(n)
    posterior_mu = mu1
    inv_sigma22 = np.linalg.pinv(sigma22) if np.linalg.det(sigma22) == 0 else np.linalg.inv(sigma22)
    posterior_sigma = sigma11 - sigma12 @ inv_sigma22 @ sigma21
    return posterior_mu, posterior_sigma


def q1_1(d, n, mu=None, sigma=None):
    """
    Generate samples from multivariate normal distribution with zero mean and identity covariance
    """
    mu = np.zeros(d) if mu is None else mu
    sigma = np.eye(d) if sigma is None else sigma
    return simulate(mu, sigma, n)


def q1_2(d, n, mu=None, sigma=None):
    """
    Generate samples from multivariate normal distribution with zero mean and all-ones covariance
    """
    mu = np.zeros(d) if mu is None else mu
    sigma = np.ones((d, d)) if sigma is None else sigma
    return simulate(mu, sigma, n)


def q2(X, n):
    """
    Generate samples from multivariate normal distribution with exponential kernel covariance
    """
    d = X.shape[0]
    mu = np.zeros(d)
    sigma = kernel_matrix(X.reshape(d, -1), exp_kernel)
    return simulate(mu, sigma, n)


def q3(X, n):
    """
    Generate samples from multivariate normal distribution with periodic kernel covariance
    """
    d = X.shape[0]
    mu = np.zeros(d)
    sigma = kernel_matrix(X.reshape(d, -1), periodic_kernel)
    return simulate(mu, sigma, n)


def q4(X, X_train, Y_train, n):
    """
    Generate samples from the posterior distribution of the Gaussian process with exponential kernel covariance
    """
    mu, sigma = posterior_inference(X, X_train, Y_train, exp_kernel)
    return simulate(mu, sigma, n)


def q5(X, X_train, Y_train, n):
    """
    Generate samples from the posterior distribution of the Gaussian process with periodic kernel covariance
    """
    mu, sigma = posterior_inference(X, X_train, Y_train, periodic_kernel)
    return simulate(mu, sigma, n)


def visualization(x, Y, title):
    """
    Visualize the generated functions
    """
    plt.clf()
    plt.title(title)
    for i in range(n):
        plt.plot(x, Y[i, :], '-', markersize=0.5, color=colors[i])
        plt.xlim([-10, 10])
        plt.ylim([-3, 3])
    return plt


def visualization_posterior(X, Y, X_train, Y_train, title):
    """
    Visualize the generated functions with training data
    """
    plt.clf()
    plt.title(title)
    for i in range(n):
        plt.plot(X, Y[i, :], '-', markersize=0.5, color=colors[i])
        plt.xlim([-10, 10])
        plt.ylim([-3, 3])
    plt.plot(X_train, Y_train, 'x', color='violet', markersize=8)
    plt.plot(X, Y.mean(axis=0), 'k--', markersize=0.5)
    return plt


if __name__ == '__main__':
    # Set the parameters
    n = 4
    d = 500
    x = np.linspace(-10, 10, d)
    colors = ["tomato", "skyblue", "lightgreen", "orange"]

    #####
    # q1-1
    title = "Functions with Identity Covariance for (v)"
    Y = q1_1(d, n)
    # visualization(x, Y, title).show()
    visualization(x, Y, title).savefig("output/hw4-2-v-1.png")

    #####
    # q1-2
    title = "Functions with All-ones Covariance for (v)"
    Y = q1_2(d, n)
    # visualization(x, Y, title).show()
    visualization(x, Y, title).savefig("output/hw4-2-v-2.png")

    #####
    # q2
    Y = q2(x, n)
    title = "Functions with Exponential Kernel Covariance for (vi)"
    # visualization(x, Y, title).show()
    visualization(x, Y, title).savefig("output/hw4-2-vi.png")

    #####
    # q3
    Y = q3(x, n)
    title = "Periodic Functions for (vii)"
    # visualization(x, Y, title).show()
    visualization(x, Y, title).savefig("output/hw4-2-vii.png")

    #####
    # Set the training data
    X_train = np.array([-6, 0, 7])
    Y_train = np.array([3, -2, 2])

    #####
    # q4
    Y = q4(x, X_train, Y_train, n)
    title = "Posterior Functions for (ix)"
    # visualization_posterior(x, Y, X_train, Y_train, title).show()
    visualization_posterior(x, Y, X_train, Y_train, title).savefig("output/hw4-2-ix.png")

    #####
    # q5
    Y = q5(x, X_train, Y_train, n)
    title = "Posterior Functions for (x)"
    # visualization_posterior(x, Y, X_train, Y_train, title).show()
    visualization_posterior(x, Y, X_train, Y_train, title).savefig("output/hw4-2-x.png")
