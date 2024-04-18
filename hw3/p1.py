import numpy as np
import matplotlib.pyplot as plt


def simulate(d, n=10000, mode='normal'):
    mu = np.zeros(d)
    sigma = np.eye(d)
    if mode == 'normal':
        data = np.random.multivariate_normal(mu, sigma, n)
    elif mode == 'uniform':
        data = np.random.uniform(-1, 1, (n, d))
    squared_distance = np.linalg.norm(data, axis=1) ** 2
    return squared_distance, np.mean(squared_distance)


if __name__ == '__main__':
    dims = [1, 2, 3, 5, 10, 50, 100]
    # mode = 'uniform'
    mode = 'normal'
    nrow = 3
    ncol = len(dims) // nrow + 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(10, 8))
    plt.suptitle(f'Squared Distance Distribution ({mode.title()})')
    for i, ax in enumerate(axs.flat):
        if i < len(dims):
            d = dims[i]
            dist, mean = simulate(d, mode=mode)
            ylim = 1.2 * max(np.histogram(dist, bins=50)[0])
            xlim = max(dist)
            ax.set_title(f'd = {d}')
            ax.hist(dist, bins=50, alpha=0.9, color='steelblue')
            if mode == 'normal':
                ax.plot([d, d], [0, ylim], color='black', linewidth=1)
            elif mode == 'uniform':
                ax.plot([d / 3, d / 3], [0, ylim], color='black', linewidth=1)
            ax.plot([mean, mean], [0, ylim], color='tomato', linestyle='dashed', linewidth=0.75)
            ax.set_ylim([0, ylim])
            ax.set_xlim([0, xlim])
        else:
            ax.axis('off')
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.savefig(f"output/hw3-{mode}.png")
    # plt.show()

