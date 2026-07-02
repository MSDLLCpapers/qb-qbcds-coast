#!/usr/bin/env python

# Load libraries
import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
from multiprocessing import Pool
import time
import psutil
import threading
import pickle
import argparse

#from rpy2.robjects import conversion, default_converter, pandas2ri
#from rpy2.robjects.conversion import localconverter
#from rpy2.robjects.packages import importr
#import rpy2.robjects as robjects

import os
import sys
# py-tetrad is not bundled with COAST; point at the user's local checkout
# via the PYTETRAD_HOME environment variable (see INSTALL.txt).
sys.path.insert(1, os.path.join(os.environ['PYTETRAD_HOME'], 'pytetrad'))

from coast.simulation import *
#from coast.feature_selection import *
#from coast.classify_variable_relationships import *
#from coast.causal_graph_gcastle import *
from coast.causal_model import *
from coast.distribution_change_new import *
from coast.optimization import *
from coast.optimization_parallelization import *
#from coast.pathway_analysis import *
from coast.utils import *

import matplotlib.pyplot as plt
import dowhy.utils.plotting as dowhy_plot
import logging
import warnings
warnings.filterwarnings("ignore")

import random

print('Loading libraries completed.')

monitoring_active = False
def monitor_process(interval=1):
    """Monitors the process in a separate thread."""
    global monitoring_active
    while monitoring_active:
        main_process = psutil.Process(os.getpid())
        children = main_process.children(recursive=True)
        all_processes = [main_process] + children

        cpu_usage = sum(p.cpu_percent(interval=0.1) for p in all_processes)
        memory_usage_gb = sum(p.memory_info().rss for p in all_processes) / (1024**3)
        memory_usage_percent = (memory_usage_gb / (psutil.virtual_memory().total / (1024**3))) * 100

        logging.info(f"[MONITOR] CPU: {cpu_usage:.2f}% | Memory: {memory_usage_gb:.2f} GB ({memory_usage_percent:.2f}%)")
        time.sleep(interval)


def main(n_nodes,
         edge_prob,
         noise_level,
         noise_type,
         a_size,
         save_dir,
         seed,
         num_interventions,
         max_combinations,
         interval,
        ):
    global monitoring_active

    np.random.seed(seed)
    random.seed(seed)
    savedir = os.path.join(
		args.save_dir,
		'nnodes{}_prob{:.2f}_noise{:.2f}{}_ntargets{}_true'.format(n_nodes, edge_prob, noise_level, noise_type, a_size),
        'seed{}'.format(seed)
		)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    logging.basicConfig(
        filename="{}/run.log".format(savedir),
        filemode="w",
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Generate synthetic data
    if n_nodes>100:
        W = generate_W(d=n_nodes, prob=edge_prob, low=0.4, high=1.0)
    else:
        W = generate_W(d=n_nodes, prob=edge_prob)
    c = np.zeros(n_nodes)
    s = np.ones([n_nodes]) * noise_level
    cov = np.identity(n_nodes) * noise_level
    A = np.linalg.solve(np.identity(n_nodes) - W, np.identity(n_nodes))
    scale = np.matmul(np.matmul(A, np.identity(n_nodes)), A.T).diagonal().reshape(-1,1)**(-0.5)
    s_new = (scale * cov * (scale.reshape(-1))).diagonal()
    W_new = scale * W * ((1/scale).reshape(-1))
    data, b_, c_ = gen_data_given_model(W_new, s_new, c, n_samples=2000, noise_type=noise_type, permutate=False)

    iv_nodes = np.random.choice(n_nodes, a_size, replace=False)
    sgn = np.random.choice(2, a_size)*2-1
    iv_vals = np.random.uniform(0.5, 2, a_size) * sgn
    a = np.zeros(n_nodes)
    for i in range(a_size):
        a[iv_nodes[i]] = iv_vals[i]
    np.savetxt('{}/ground_truth_interventions.txt'.format(savedir), a)
    data_new = gen_new_data_given_model(b_, s_new, c, a, n_samples=2000, noise_type=noise_type, permutate=False)

    data.to_csv('{}/data.csv'.format(savedir), index=False)
    data_new.to_csv('{}/data_new.csv'.format(savedir), index=False)

    logging.info('Data generated and saved')

    g_true = nx.DiGraph(get_edges_simulation(W))
    for col in data.columns:
        if col not in g_true.nodes():
            g_true.add_node(col)
    with open('{}/ground_truth_dag.pkl'.format(savedir), 'wb') as file:
        pickle.dump(g_true, file)

    # Learn causal graph structure
    m = gcm.StructuralCausalModel(g_true)
    custom_mechanisms(g_true, m)
    gcm.fit(m, data)

    with open('{}/fitted_model_source.pkl'.format(savedir), 'wb') as file:
        pickle.dump(m, file)

    logging.info('Causal graph and model fitted and saved')

    # Initialization
    causal_object = CausalClass(data_old=data, data_new=data_new, causal_graph=g_true, causal_model=m)
    causal_object.get_total_changes()
    causal_object.fit_old_model_on_new_data()
    mechanism_changes = causal_object.check_mechanism_changes()
    with open('{}/mechanism_changes.pkl'.format(savedir), 'wb') as file:
        pickle.dump(mechanism_changes, file)

    logging.info('Mechanism changes saved')

    monitoring_active = True
    monitor_thread = threading.Thread(target=monitor_process, args=(interval,))
    monitor_thread.daemon = True
    monitor_thread.start()

    ### Approach 1 (modified from dowhy, very quick)
    attrib_scores_est = estimate_distribution_change_scores_optimized(m, data, data_new, mechanism_changed_for_node=mechanism_changes, difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x))
    attrib_scores_est = sort_nodes(attrib_scores_est)
    attrib_scores_est = {k: abs(v) for k, v in attrib_scores_est.items()}
    attrib_scores_est_for_plot = dict(list(attrib_scores_est.items())[:20])
    dowhy_plot.bar_plot(attrib_scores_est_for_plot, ylabel='Estimated distribution change attribution score', display_plot=False)
    plt.tight_layout()
    plt.savefig('{}/attrib_scores_est.png'.format(savedir))
    plt.close()
    with open('{}/attrib_scores_est.pkl'.format(savedir), 'wb') as file:
        pickle.dump(attrib_scores_est, file)

    logging.info('Attribution scores (approach 1) completed and results saved')

    ### Approach 2 (accumulative attribution scores)
    attrib_scores = get_attribution_scores_parallel_standalone(causal_object, de_cols=None, test='parametric', alpha=0.05)
    attrib_scores_for_plot = dict(list(attrib_scores.items())[:20])
    dowhy_plot.bar_plot(attrib_scores_for_plot, ylabel='Accumulative distribution change attribution score', display_plot=False)
    plt.tight_layout()
    plt.savefig('{}/attrib_scores.png'.format(savedir))
    plt.close()
    with open('{}/attrib_scores.pkl'.format(savedir), 'wb') as file:
        pickle.dump(attrib_scores, file)

    logging.info('Attribution scores (approach 2) completed and results saved')

    # Get candidate nodes
    top_nodes, normalized_attributions = causal_object.adaptive_selection(attrib_scores, delta=0.01)
    logging.info(f'Adaptive selection of top nodes completed, {len(top_nodes)} nodes were selected')

    # Optimize by combinatorial approach
    ## first run interventions on every single node
    solutions_single = get_perturbation_targets_by_comb_parallel_standalone(causal_object, top_nodes=top_nodes, attrib_scores=attrib_scores, num_interventions=1, max_combinations=max_combinations, minval=-5000, maxval=5000, maxiter=10000, minimize_target_state_change=False, gamma=1e-2)
    solutions_single['Node'] = solutions_single['intervention targets'].map(lambda x: x[0])
    transprop_single = dict(zip(solutions_single['Node'], solutions_single['TransProp']))
    solutions_single.to_csv('{}/solutions_single.csv'.format(savedir), index=False)
    logging.info('Single interventions of top nodes completed')
    ## then use their transition scores as the new scores for ranking top combinations
    solutions_comb = get_perturbation_targets_by_comb_parallel_standalone(causal_object, top_nodes=top_nodes, attrib_scores=transprop_single, num_interventions=num_interventions, max_combinations=max_combinations, minval=-5000, maxval=5000, maxiter=10000, minimize_target_state_change=False, gamma=1e-2)
    solutions_comb.to_csv('{}/solutions_comb.csv'.format(savedir), index=False)
    logging.info('Combinatorial solutions saved')

    # Optimize by regularization approach
    solutions_reg = get_perturbation_targets_by_reg_parallel_standalone(causal_object, top_nodes=top_nodes, attrib_scores=attrib_scores, minval=-5000, maxval=5000, maxiter=10000, minimize_target_state_change=False, gamma=1e-2)
    solutions_reg.to_csv('{}/solutions_reg.csv'.format(savedir), index=False)
    solutions_reg_processed = causal_object.process_results_by_reg(solutions_reg, thres_val=1e-1)
    solutions_reg_processed.to_csv('{}/solutions_reg_processed.csv'.format(savedir), index=False)

    plot_df = solutions_reg_processed.sort_values(by='Lambda')
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(plot_df['Lambda'], plot_df['# of intervention targets'], marker='o')
    axs[0].set_ylabel('Number of Targets')
    axs[0].set_xlabel('Regularization Parameter Value')
    axs[1].plot(plot_df['Lambda'], plot_df['TransProp'], marker='o')
    axs[1].set_ylabel('Transition Score')
    axs[1].set_xlabel('Regularization Parameter Value')
    plt.savefig(f'{savedir}/reg_top_{len(top_nodes)}_nodes.png')
    plt.close()

    logging.info('Regularization solutions completed and saved')

    # Supplementary: DE-based intervention targets
    DE_res = post_DE(data, data_new)
    DE_res.to_csv('{}/DE_results.csv'.format(savedir))
    size_list = list(solutions_reg_processed['# of intervention targets'].unique())
    comb_list = []
    for size in size_list:
        if size > 0:
            comb_list.append(list(DE_res.index[:int(size)]))
    solutions_DE_supp = get_perturbation_targets_by_specified_combs_parallel_standalone(causal_object, combs=comb_list, minval=-5000, maxval=5000, maxiter=10000, minimize_target_state_change=False, gamma=1e-2)
    solutions_DE_supp.to_csv('{}/solutions_DE_supp.csv'.format(savedir), index=False)

    logging.info('DE supplementary solutions saved')

    monitoring_active = False
    monitor_thread.join()
    print("Monitoring ceased.")

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_nodes', type=int, help='<Required> Number of nodes in the graph (int).', required=True)
    parser.add_argument('--edge_prob', type=float, help='<Required> Edge probability of the graph (float).', required=True)
    parser.add_argument('--noise_level', type=float, help='<Required> Noise level of observations (float).', required=True)
    parser.add_argument('--noise_type', type=str, default='gaussian', help='Noise type, gaussian or lingam.')
    parser.add_argument('--a_size', type=int, help='<Required> Number of intervention targets (int).', required=True)
    parser.add_argument('--save_dir', type=str, help="<Required> Saving directory.")
    parser.add_argument('--seed', type=int, default=88, help="Random seed.")
    parser.add_argument('--log_interval', type=int, default=1, help="Interval time to log CPU and Memory usage.")

	# read args
    args = parser.parse_args()

    num_interventions=3
    max_combinations=500

    main(args.n_nodes, args.edge_prob, args.noise_level, args.noise_type, args.a_size, args.save_dir, args.seed,
         num_interventions, max_combinations, args.log_interval)
