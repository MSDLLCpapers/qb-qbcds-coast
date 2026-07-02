from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import GOLEM, DirectLiNGAM, ICALiNGAM


# GOLEM
# See https://github.com/huawei-noah/trustworthyAI/blob/master/gcastle/castle/algorithms/gradient/notears/torch/golem.py
def run_golem(X, learning_rate=1e-3, num_iter=1e5, thres=0.3, device_type='cpu'):
    g = GOLEM(learning_rate=learning_rate, num_iter=num_iter, graph_thres=thres, device_type=device_type)
    g.learn(X)
    return g


# DirectLiNGAM
def run_direct_lingam(X, thres=0.3):
    n = DirectLiNGAM(thresh=thres)
    n.learn(X)
    return n


# ICA-LiNGAM
def run_ica_lingam(X, thres=0.3):
    n = ICALiNGAM(thresh=thres)
    n.learn(X)
    return n