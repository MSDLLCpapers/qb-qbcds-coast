import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import math
import io
import os
import sys

from dowhy import gcm
from dowhy.gcm.falsify import FalsifyConst, falsify_graph, plot_local_insights, run_validations, apply_suggestions
from dowhy.gcm.independence_test.generalised_cov_measure import generalised_cov_based
from dowhy.gcm.util import plot
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.ml import SklearnRegressionModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import norm, ttest_ind, mannwhitneyu


#def get_edges(filename):
#    """Return a list of edges from the output of causal discovery algorithms, which will be used to construct a DAG."""
#    
#    edges = []
#    with open(filename, "r") as file:
#        for line in file.readlines():
#            line = line.strip()
#            if '-->' in line:
#                words = line.split('-->')
#                temp = words[0].split('.')
#                node1 = temp[1][1:-1]
#                node2 = words[1][1:]
#                edges.append((node1, node2))
#    return edges

def get_edges(filename):
    """Return a list of edges from the output of causal discovery algorithms, which will be used to construct a DAG.
    Modified to handle gene names with "." properly.
    """
    edges = []
    with open(filename, "r") as file:
        for line in file.readlines():
            line = line.strip()
            if '-->' in line:
                words = line.split('-->')
                part1 = words[0]
                dot_idx = part1.index(".")
                node1 = part1[(dot_idx+2):-1]
                node2 = words[1][1:]
                edges.append((node1, node2))
    return edges


def get_edges_from_matrix(adj_matrix, var_names):
    """
    get graph edges from an adjacency matrix
    """
    d = adj_matrix.shape[0]
    edges = []
    for i in range(d):
        for j in range(d):
            if abs(adj_matrix[i,j])>0:
                edges.append((var_names[i], var_names[j]))
    return edges


def is_dag_by_matrix(B):
    """
    Check whether B corresponds to a DAG.

    Parameters
    ----------
    B: numpy.ndarray
        [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))


def custom_mechanisms(causal_graph, causal_model):
    """
    assign causal mechanisms to each node
    if root node, set a scipy distribution, otherwise a linear mechanism
    """
    node_list = list(causal_graph.nodes)
    for i in tqdm(range(len(node_list))):
        node = node_list[i]
        if causal_graph.in_degree(node)==0:
            causal_model.set_causal_mechanism(node, gcm.ScipyDistribution(norm))
        else:
            causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(prediction_model=gcm.ml.create_linear_regressor(), noise_model=gcm.ScipyDistribution(norm)))


##### Efficient Casual Model Evaluation
# Define independence test based on the generalised covariance measure with gradient boosted decision trees as models
def create_gradient_boost_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(GradientBoostingRegressor(**kwargs))
def custom_gbr_indep_test(X, Y, Z=None):
    return generalised_cov_based(X, Y, Z=Z, prediction_model_X=create_gradient_boost_regressor,
                                 prediction_model_Y=create_gradient_boost_regressor)

def create_linear_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(LinearRegression(**kwargs))
def custom_linear_indep_test(X, Y, Z=None):
    return generalised_cov_based(X, Y, Z=Z, prediction_model_X=create_linear_regressor,
                                 prediction_model_Y=create_linear_regressor)

def eval_graph(g, data, n_permutations=100, model='linear'):
    set_random_seed(2222)
    if model=='linear':
        result = falsify_graph(g, data, n_permutations=n_permutations, independence_test=custom_linear_indep_test, conditional_independence_test=custom_linear_indep_test, plot_histogram=True, suggestions=True)
    elif model=='gbr':
        result = falsify_graph(g, data, n_permutations=n_permutations, independence_test=custom_gbr_indep_test, conditional_independence_test=custom_gbr_indep_test, plot_histogram=True, suggestions=True)
    print(result)
    return result