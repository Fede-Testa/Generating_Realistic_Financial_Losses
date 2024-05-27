import numpy as np
import torch


def AD_distance(X_og, X_gen):
    """
    Calculates the Anderson-Darling distance between two datasets.

    The Anderson-Darling distance is a statistical measure used to compare the similarity
    between two datasets. It measures the discrepancy between the empirical distribution
    function of the original dataset (X_og) and the empirical distribution function of the
    generated dataset (X_gen), with a focus on the tails of the distribution.

    Parameters:
    - X_og (array-like): The original dataset.
    - X_gen (array-like): The generated dataset.

    Returns:
    - float: The Anderson-Darling distance between the two datasets.
    """
    if isinstance(X_og, torch.Tensor):
        X_og = X_og.data.numpy()
    if isinstance(X_gen, torch.Tensor):
        X_gen = X_gen.data.numpy()

    X_og = np.array(X_og)
    X_gen = np.array(X_gen)
    n = X_og.shape[0]
    d = X_og.shape[1]

    u_tilde = np.array([mod_prob(X_og[:, i], X_gen[:, i], n) for i in range(d)])

    weigth = np.array([(2*i + 1) for i in range(n)])
    W = - n - np.mean(
        (weigth*(np.log(u_tilde) + np.log(1 - u_tilde[:, ::-1]))), axis=1
        )

    return np.mean(W)


def mod_prob(v_og, v_gen, n):
    """
    Auxiliary computation for the Anderson-Darling distance.

    Parameters:
    v_og (array-like): The original values.
    v_gen (array-like): The generated values.
    n (int): The number of values.

    Returns:
    array-like: The modified probabilities.

    """
    v_gen = np.sort(v_gen)
    u_tilde = np.array([sum(v_og <= v_gen[i]) for i in range(n)])
    return (u_tilde+1)/(n+2)


def Absolute_Kendall_error(X_og, X_gen):
    """
    Calculate the absolute Kendall error between two sets of data.

    Parameters:
    X_og (array-like): The original data.
    X_gen (array-like): The generated data.

    Returns:
    float: The absolute Kendall error.
    """
    if isinstance(X_og, torch.Tensor):
        X_og = X_og.data.numpy()
    if isinstance(X_gen, torch.Tensor):
        X_gen = X_gen.data.numpy()
        
    Z_og = pseudo_obs(np.array(X_og))
    Z_gen = pseudo_obs(np.array(X_gen))
    return np.linalg.norm(np.sort(Z_og) - np.sort(Z_gen), 1) / Z_og.shape[0]


def pseudo_obs(x):
    """
    Auxiliary function to compute the Absolute Kendall error.
    It computes the pseudo-observations of the data.

    Parameters:
    - x (array-like): The input data.

    Returns:
    - array-like: The pseudo-observations of the data.
    """
    n = x.shape[0]
    d = x.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j == i:
                M[i, i] = False
            else:
                bool_val = True
                for c in range(d):
                    if x[j, c] >= x[i, c]:
                        bool_val = False
                M[i, j] = bool_val

    return np.sum(M, 0) / (n - 1)


def energy_distance(x, y):
    """
    Calculates the (empirical) energy distance between two sets of points.
    Computations are in torch to ensure automatic differentiation.
    If either input is not a tensor, returns a numpy scalar.

    Parameters:
    x (torch.Tensor): The first set of points, with shape (n, d).
    y (torch.Tensor): The second set of points, with shape (m, d).

    Returns:
    float (possibly tensor with grad): The energy distance between the two sets of points.
    """
    to_numpy = False
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
        to_numpy = True
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
        to_numpy = True

    n = x.shape[0]
    m = y.shape[0]

    x_reshaped1 = torch.unsqueeze(x, 1)
    y_reshaped2 = torch.unsqueeze(y, 0)
    x_reshaped2 = torch.unsqueeze(x, 0)
    y_reshaped1 = torch.unsqueeze(y, 1)

    normsA = torch.linalg.norm(x_reshaped1 - y_reshaped2, axis=2)
    A = torch.sum(normsA)
    normsB = torch.linalg.norm(x_reshaped1 - x_reshaped2, axis=2)
    B = torch.sum(normsB)
    normsC = torch.linalg.norm(y_reshaped1 - y_reshaped2, axis=2)
    C = torch.sum(normsC)
    distance = 2*A/(n*m) - C/(m**2) - B/(n**2)
    if to_numpy:
        return distance.data.numpy()
    return distance
