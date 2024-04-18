# utils functions for neural network training
import numpy as np

from nn import Loss, Network, Optimizer, DataLoader


def convert_img(X: np.array, Y: np.array) -> np.array:
    """
    Convert the given X and Y to an image.
    """
    h, w, c = int(np.max(X[:, 0])), int(np.max(X[:, 1])), Y.shape[-1]
    img = np.zeros((h, w, c), dtype=np.uint8)
    for i in range(X.shape[0]):
        img[int(X[i, 0]) - 1, int(X[i, 1]) - 1, :] = Y[i]
    return img


def train(
        model: Network,
        dataloader: DataLoader,
        optimizer: Optimizer,
        loss_fn: Loss,
        num_iterations: int,
        verbose: bool = False
) -> list[float]:
    """
    Train the model using the given data and hyperparameters.
    """
    # TODO: repeatedly do forward and backwards calls, update weights, do
    # stochastic gradient descent on mini-batches.
    size = len(dataloader)
    epoch_loss = []
    for i in range(num_iterations):
        print(f"Epoch {i+1}\n-------------------------------")
        for batch, (data, labels) in enumerate(dataloader):
            output = model.forward(data)
            loss = loss_fn.forward(output, labels)
            gradient = loss_fn.backward()
            model.backward(gradient)
            optimizer.step()
            optimizer.zero_grad()
            if verbose and batch % 1000:
                print(f"Batch loss: {loss:>7f}  [{(batch+1) * len(data):>5d}/{size:>5d}]")
        epoch_loss.append(loss)
    return epoch_loss


# def train_test_split(X: np.array, y: np.array, test_ratio: float, shuffle: bool = False):
#     """
#     Split dataset
#     """
#     n = X.shape[0]
#     idx = np.arange(n)
#     if shuffle:
#         np.random.shuffle(idx)
#     test_size: int = int(np.round(n * test_ratio, 0))
#     test_idx, train_idx = idx[:test_size], idx[test_size:]
#     return (X[train_idx],  # X_train
#             X[test_idx],  # X_test
#             y[train_idx],  # y_train
#             y[test_idx])  # y_test