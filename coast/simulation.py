# See https://github.com/huawei-noah/trustworthyAI/blob/master/datasets/README.md

import numpy as np
import pandas as pd
from copy import deepcopy

def generate_W(d=6, prob=0.5, low=0.5, high=2.0):
    """
    generate a random weighted adjacency matrix
    :param d: number of nodes
    :param prob: prob of existing an edge
    :return:
    """
    g_random = np.float32(np.random.rand(d,d)<prob)
    g_random = np.tril(g_random, -1)
    U = np.round(np.random.uniform(low=low, high=high, size=[d, d]), 1)
    U[np.random.randn(d, d) < 0] *= -1
    W = (g_random != 0).astype(float) * U
    return W


def gen_data_given_model(b, s, c, n_samples=10, noise_type='lingam', permutate=False):
    """Generate artificial data based on the given model.
       Based on ICA-LiNGAM codes.
       https://github.com/cdt15/lingam
    Parameters
    ----------
    b : numpy.ndarray, shape=(n_features, n_features)
        Strictly lower triangular coefficient matrix.
        NOTE: Each row of `b` corresponds to each variable, i.e., X = BX.
    s : numpy.ndarray, shape=(n_features,)
        Scales of disturbance variables.
    c : numpy.ndarray, shape=(n_features,)
        Means of observed variables.

    Returns
    -------
    xs, b_, c_ : Tuple
        `xs` is observation matrix, where `xs.shape==(n_samples, n_features)`.
        `b_` is permuted coefficient matrix. Note that rows of `b_` correspond
        to columns of `xs`. `c_` if permuted mean vectors.
    """
    n_vars = b.shape[0]

    # Check args
    assert (b.shape == (n_vars, n_vars))
    assert (s.shape == (n_vars,))
    assert (np.sum(np.abs(np.diag(b))) == 0)
    np.allclose(b, np.tril(b))

    if noise_type == 'lingam':
        # Nonlinearity exponent, selected to lie in [0.5, 0.8] or [1.2, 2.0].
        # (<1 gives subgaussian, >1 gives supergaussian)
        q = np.random.rand(n_vars) * 1.1 + 0.5
        ixs = np.where(q > 0.8)
        q[ixs] = q[ixs] + 0.4

        # Generates disturbance variables
        ss = np.random.randn(n_samples, n_vars)
        ss = np.sign(ss) * (np.abs(ss) ** q)

        # Normalizes the disturbance variables to have the appropriate scales
        ss = ss / np.std(ss, axis=0) * s

    elif noise_type == 'gaussian':
        ss = np.random.randn(n_samples, n_vars) * s

    # Generate the data one component at a time
    xs = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        # NOTE: columns of xs and ss correspond to rows of b
        xs[:, i] = ss[:, i] + xs.dot(b[i, :]) + c[i]

        # Permute variables
    b_ = deepcopy(b)
    c_ = deepcopy(c)
    if permutate:
        p = np.random.permutation(n_vars)
        xs[:, :] = xs[:, p]
        b_[:, :] = b_[p, :]
        b_[:, :] = b_[:, p]
        c_[:] = c[p]

    xs = pd.DataFrame(xs)
    xs.columns = [f'X{i}' for i in range(1, n_vars+1)]

    return xs, b_, c_


def gen_new_data_given_model(b, s, c, a, n_samples=10, noise_type='lingam', permutate=False):
    """
    Parameters
    ----------
    a : numpy.ndarray, shape=(n_features,)
        Shift intervention values.
    """
    n_vars = b.shape[0]

    # Check args
    assert (b.shape == (n_vars, n_vars))
    assert (s.shape == (n_vars,))
    assert (np.sum(np.abs(np.diag(b))) == 0)
    np.allclose(b, np.tril(b))

    if noise_type == 'lingam':
        # Nonlinearity exponent, selected to lie in [0.5, 0.8] or [1.2, 2.0].
        # (<1 gives subgaussian, >1 gives supergaussian)
        q = np.random.rand(n_vars) * 1.1 + 0.5
        ixs = np.where(q > 0.8)
        q[ixs] = q[ixs] + 0.4
        
        # Generates disturbance variables
        ss = np.random.randn(n_samples, n_vars)
        ss = np.sign(ss) * (np.abs(ss) ** q)

        # Normalizes the disturbance variables to have the appropriate scales
        ss = ss / np.std(ss, axis=0) * s

    elif noise_type == 'gaussian':
        ss = np.random.randn(n_samples, n_vars) * s

    # Generate the data one component at a time
    xs = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        # NOTE: columns of xs and ss correspond to rows of b
        xs[:, i] = ss[:, i] + xs.dot(b[i, :]) + c[i] + a[i]

        # Permute variables
    b_ = deepcopy(b)
    c_ = deepcopy(c)
    if permutate:
        p = np.random.permutation(n_vars)
        xs[:, :] = xs[:, p]
        b_[:, :] = b_[p, :]
        b_[:, :] = b_[:, p]
        c_[:] = c[p]

    xs = pd.DataFrame(xs)
    xs.columns = [f'X{i}' for i in range(1, n_vars+1)]

    return xs


def get_edges_simulation(coeff_matrix):
    """
    get graph edges from a lower triangular adjacency matrix
    (specifically for simulation.py)
    """
    d = coeff_matrix.shape[0]
    names = [f'X{i}' for i in range(1, d+1)]
    edges = []
    for i in range(d):
        for j in range(i):
            if abs(coeff_matrix[i,j])>1e-3:
                edges.append((f'X{j+1}', f'X{i+1}'))
    return edges
    

####### Example usage (10 variables, 2 interventions):
"""
seed = 8
np.random.seed(seed)

d = 10
W = generate_W(d=d, prob=0.25)
c = np.zeros(d)
s = np.ones([d])

A = np.linalg.inv(np.identity(d) - W)
scale = np.matmul(np.matmul(A, np.identity(d)), A.T).diagonal().reshape(-1,1)**(-0.5)
s_new = (scale * np.identity(d) * (scale.reshape(-1))).diagonal()
W_new = scale * W * ((1/scale).reshape(-1))

data, b_, c_ = gen_data_given_model(W_new, s_new, c, n_samples=2000, noise_type='gaussian', permutate=False)

iv_num = 2
iv_nodes = np.random.choice(d, iv_num, replace=False)
iv_vals = np.random.uniform(0.5, 2, iv_num)
print(iv_nodes)
print(iv_vals)
a = np.zeros(d)
for i in range(iv_num):
    a[iv_nodes[i]] = iv_vals[i]

data_new = gen_new_data_given_model(b_, s_new, c, a, n_samples=2000, noise_type='gaussian', permutate=False)
"""