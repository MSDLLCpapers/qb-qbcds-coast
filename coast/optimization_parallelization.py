import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
import itertools
from scipy.optimize import minimize
from tqdm import tqdm
from multiprocessing import Pool
from joblib import Parallel, delayed
import os
import warnings
from typing import Optional, List, Dict, Tuple, Any
from coast.utils import *


def get_perturbation_targets_by_comb_parallel_standalone(
    causal_object,
    top_nodes: List[Any],
    attrib_scores: Dict[Any, float],
    num_interventions: int = 3,
    max_combinations: int = 500,
    minval: float = 0,
    maxval: float = 10000,
    maxiter: int = 10000,
    minimize_target_state_change: bool = False,
    gamma: float = 1e-2,
    nodes_need_constancy: Optional[List[Any]] = None,
    thres_change: float = 0.5,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    verbose: int = 1,
    bounds_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Find optimal perturbation targets using parallel optimization.

    Parameters
    ----------
    causal_object : CausalObject
        The causal analysis object
    top_nodes : List[Any]
        List of candidate nodes for intervention
    attrib_scores : Dict[Any, float]
        Attribution scores for each node
    num_interventions : int
        Number of nodes to intervene on simultaneously
    max_combinations : int
        Maximum number of combinations to evaluate
    minval : float
        Minimum intervention value
    maxval : float
        Maximum intervention value
    maxiter : int
        Maximum iterations for optimization
    minimize_target_state_change : bool
        Whether to minimize target state change
    gamma : float
        Regularization parameter
    nodes_need_constancy : Optional[List[Any]]
        Nodes that should remain constant
    thres_change : float
        Threshold for change
    n_jobs : Optional[int]
        Number of parallel jobs (None = auto-detect)
    backend : str
        Joblib backend ('loky', 'threading', 'multiprocessing')
    verbose : int
        Verbosity level (0=silent, 10=progress)
    bounds_type : Optional[str]
        None (no bounds), "positive" (all >= 0), or "negative" (all <= 0)

    Returns
    -------
    pd.DataFrame
        Results with columns: intervention targets, intervention values, Diff, TransProp
    """
    # Auto-detect CPUs (PBS-aware)
    if n_jobs is None:
        n_jobs = int(os.environ.get("NCPUS", os.cpu_count()))
    
    print(f"Using {n_jobs} workers for parallel optimization")
    
    # Generate and rank combinations efficiently
    print(f"Generating combinations of {num_interventions} nodes from {len(top_nodes)} candidates...")
    top_combs = _get_top_combinations(
        top_nodes, 
        attrib_scores, 
        num_interventions, 
        max_combinations
    )
    
    print(f"Evaluating {len(top_combs)} combinations...")
    
    # Pre-compute shared data to avoid serialization overhead
    shared_params = {
        'minval': minval,
        'maxval': maxval,
        'maxiter': maxiter,
        'minimize_target_state_change': minimize_target_state_change,
        'gamma': gamma,
        'nodes_need_constancy': nodes_need_constancy,
        'thres_change': thres_change,
        'bounds_type': bounds_type
    }

    # Run parallel optimization
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_process_single_combination)(
            causal_object, comb, shared_params
        )
        for comb in top_combs
    )
    
    # Filter and format results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("Warning: No valid results found!")
        return pd.DataFrame(columns=['intervention targets', 'intervention values', 'Diff', 'TransProp'])
    
    print(f"Successfully optimized {len(valid_results)}/{len(top_combs)} combinations")
    
    # Create output DataFrame
    outputs = pd.DataFrame(
        valid_results, 
        columns=['intervention targets', 'intervention values', 'Diff']
    )
    outputs['TransProp'] = 1 - (outputs['Diff'] / causal_object.total)
    
    return outputs.sort_values(by='Diff', ignore_index=True)


def _get_top_combinations(
    top_nodes: List[Any],
    attrib_scores: Dict[Any, float],
    num_interventions: int,
    max_combinations: int
) -> List[Tuple]:
    """
    Generate and rank combinations by attribution scores.
    
    Optimized to avoid creating all combinations if not needed.
    """
    # Generate combinations
    combs = itertools.combinations(top_nodes, num_interventions)
    
    # Score and sort combinations efficiently
    # Use a list of tuples for faster sorting
    scored_combs = [
        (comb, sum(attrib_scores[node] for node in comb))
        for comb in combs
    ]
    
    # Sort by score (descending) and take top max_combinations
    scored_combs.sort(key=lambda x: x[1], reverse=True)
    top_combs = [comb for comb, score in scored_combs[:max_combinations]]
    
    return top_combs


def _process_single_combination(
    causal_object,
    comb: Tuple,
    params: Dict[str, Any]
) -> Optional[Tuple]:
    """
    Process a single combination - defined at module level for better pickling.
    
    Parameters
    ----------
    causal_object : CausalObject
        The causal analysis object
    comb : Tuple
        Combination of nodes to intervene on
    params : Dict
        Shared parameters for optimization
    
    Returns
    -------
    Optional[Tuple]
        (combination, intervention_values, objective_value) or None if failed
    """
    # Suppress warnings per worker
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="X has feature names")
    
    try:
        # Extract parameters
        minval = params['minval']
        maxval = params['maxval']
        maxiter = params['maxiter']
        minimize_target_state_change = params['minimize_target_state_change']
        gamma = params['gamma']
        nodes_need_constancy = params['nodes_need_constancy']
        thres_change = params['thres_change']
        bounds_type = params.get('bounds_type', None)
        
        # Define objective function
        def objective(x):
            return causal_object.objective_function_comb(
                x,
                nodes_to_intervene=comb,
                minimize_target_state_change=minimize_target_state_change,
                gamma=gamma
            )
        
        # Define constraints
        def constraint(x):
            return causal_object.define_constraints(
                x,
                nodes_to_intervene=comb,
                minval=minval,
                maxval=maxval,
                nodes_need_constancy=nodes_need_constancy,
                thres_change=thres_change
            )
        
        # Initial guess (zeros)
        x0 = np.zeros(len(comb))

        # Constraint specification
        constraints = {'type': 'ineq', 'fun': constraint}

        # Compute bounds
        bounds = causal_object.define_bounds(comb, bounds_type=bounds_type)

        # Run optimization
        result = minimize(
            objective,
            x0,
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': maxiter}
        )
        
        # Post-process result
        if minimize_target_state_change:
            sub_graph_new = causal_object.get_subgraph_new(comb)
            intervened_data_target = causal_object.get_intervened_data_target(
                result.x, comb, sub_graph_new
            )
            diff_target = gamma * np.sum(
                (intervened_data_target.mean() - causal_object.data_new.mean()) ** 2
            )
            objective_value = result.fun - diff_target
        else:
            objective_value = result.fun
        
        return (comb, result.x, objective_value)
        
    except Exception as e:
        print(f"Error processing combination {comb}: {e}")
        return None


def get_perturbation_targets_by_reg_parallel_standalone(
    causal_object,
    top_nodes: List[Any],
    attrib_scores: Dict[Any, float],
    minval: float = 0,
    maxval: float = 10000,
    maxiter: int = 10000,
    minimize_target_state_change: bool = False,
    gamma: float = 1e-2,
    nodes_need_constancy: Optional[List[Any]] = None,
    thres_change: float = 0.5,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    verbose: int = 1,
    epsilon: float = 1e-3,
    num_lambda_points: int = 20,
    min_score: float = 1e-2,
    bounds_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Find optimal perturbation targets using regularization with parallel optimization.

    Parameters
    ----------
    causal_object : CausalObject
        The causal analysis object
    top_nodes : List[Any]
        List of candidate nodes for intervention
    attrib_scores : Dict[Any, float]
        Attribution scores for each node
    minval : float
        Minimum intervention value
    maxval : float
        Maximum intervention value
    maxiter : int
        Maximum iterations for optimization
    minimize_target_state_change : bool
        Whether to minimize target state change
    gamma : float
        Regularization parameter for target state change
    nodes_need_constancy : Optional[List[Any]]
        Nodes that should remain constant
    thres_change : float
        Threshold for change
    n_jobs : Optional[int]
        Number of parallel jobs (None = auto-detect)
    backend : str
        Joblib backend ('loky', 'threading', 'multiprocessing')
    verbose : int
        Verbosity level (0=silent, 10=progress)
    epsilon : float
        Epsilon parameter for lambda range computation
    num_lambda_points : int
        Number of lambda values to evaluate
    min_score : float
        Minimum score for weight computation (prevents division by zero)
    bounds_type : Optional[str]
        None (no bounds), "positive" (all >= 0), or "negative" (all <= 0)

    Returns
    -------
    pd.DataFrame
        Results with columns: Lambda, Diff, TransProp, and intervention values for each node
    """
    # Auto-detect CPUs (PBS-aware)
    if n_jobs is None:
        n_jobs = int(os.environ.get("NCPUS", os.cpu_count()))
    
    print(f"Using {n_jobs} workers for parallel regularization optimization")
    print(f"Optimizing over {len(top_nodes)} nodes: {top_nodes}")
    
    # Compute weights efficiently
    weights = _compute_weights(top_nodes, attrib_scores, min_score)
    print(f"Computed weights (sum={np.sum(weights):.4f})")
    
    # Compute lambda range
    print(f"Computing lambda range with {num_lambda_points} points...")
    lambda_max, lambda_grid = causal_object.compute_lambda_range(
        weights=weights,
        epsilon=epsilon,
        num_points=num_lambda_points
    )
    print(f"Lambda range: [{lambda_grid[0]:.4f}, {lambda_grid[-1]:.4f}]")
    
    # Pre-compute shared parameters
    shared_params = {
        'top_nodes': top_nodes,
        'weights': weights,
        'minval': minval,
        'maxval': maxval,
        'maxiter': maxiter,
        'minimize_target_state_change': minimize_target_state_change,
        'gamma': gamma,
        'nodes_need_constancy': nodes_need_constancy,
        'thres_change': thres_change,
        'bounds_type': bounds_type
    }
    
    # Run parallel optimization
    print(f"Evaluating {len(lambda_grid)} lambda values...")
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_process_single_lambda)(
            causal_object, lambda_val, shared_params
        )
        for lambda_val in lambda_grid
    )
    
    # Filter and format results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("Warning: No valid results found!")
        return pd.DataFrame(columns=['Lambda', 'Diff', 'TransProp'] + top_nodes)
    
    print(f"Successfully optimized {len(valid_results)}/{len(lambda_grid)} lambda values")
    
    # Create output DataFrame
    outputs = pd.DataFrame(
        valid_results,
        columns=['Lambda', 'Diff', 'TransProp'] + top_nodes
    )
    
    return outputs.sort_values(by='Diff', ignore_index=True)


def _compute_weights(
    top_nodes: List[Any],
    attrib_scores: Dict[Any, float],
    min_score: float = 1e-2
) -> np.ndarray:
    """
    Compute normalized weights from attribution scores.
    
    Parameters
    ----------
    top_nodes : List[Any]
        List of nodes
    attrib_scores : Dict[Any, float]
        Attribution scores for each node
    min_score : float
        Minimum score to prevent division by zero
    
    Returns
    -------
    np.ndarray
        Normalized weights (sum to 1)
    """
    # Vectorized weight computation
    weights = np.array([
        1.0 / max(attrib_scores[node], min_score)
        for node in top_nodes
    ])
    
    # Normalize
    weights /= np.sum(weights)
    
    return weights


def _process_single_lambda(
    causal_object,
    lambda_val: float,
    params: Dict[str, Any]
) -> Optional[List]:
    """
    Process a single lambda value - defined at module level for better pickling.
    
    Parameters
    ----------
    causal_object : CausalObject
        The causal analysis object
    lambda_val : float
        Lambda regularization parameter value
    params : Dict
        Shared parameters for optimization
    
    Returns
    -------
    Optional[List]
        [lambda, diff, trans_prop, intervention_value_1, ..., intervention_value_n]
        or None if optimization failed
    """
    # Suppress warnings per worker
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="X has feature names")
    
    try:
        # Extract parameters
        top_nodes = params['top_nodes']
        weights = params['weights']
        minval = params['minval']
        maxval = params['maxval']
        maxiter = params['maxiter']
        minimize_target_state_change = params['minimize_target_state_change']
        gamma = params['gamma']
        nodes_need_constancy = params['nodes_need_constancy']
        thres_change = params['thres_change']
        bounds_type = params.get('bounds_type', None)
        
        # Define objective function
        def objective(x):
            return causal_object.objective_function_reg(
                x,
                nodes_to_intervene=top_nodes,
                Lambda=lambda_val,
                weights=weights,
                minimize_target_state_change=minimize_target_state_change,
                gamma=gamma
            )
        
        # Define constraints
        def constraint(x):
            return causal_object.define_constraints(
                x,
                nodes_to_intervene=top_nodes,
                minval=minval,
                maxval=maxval,
                nodes_need_constancy=nodes_need_constancy,
                thres_change=thres_change
            )
        
        # Initial guess (zeros)
        x0 = np.zeros(len(top_nodes))

        # Constraint specification
        constraints = {'type': 'ineq', 'fun': constraint}

        # Compute bounds
        bounds = causal_object.define_bounds(top_nodes, bounds_type=bounds_type)

        # Run optimization
        result = minimize(
            objective,
            x0,
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': maxiter}
        )
        
        # Compute penalty term
        penalty = lambda_val * np.dot(weights, np.abs(result.x))
        objective_without_penalty = result.fun - penalty
        
        # Post-process result
        if minimize_target_state_change:
            sub_graph_new = causal_object.get_subgraph_new(top_nodes)
            intervened_data_target = causal_object.get_intervened_data_target(
                result.x, top_nodes, sub_graph_new
            )
            diff_target = gamma * np.sum(
                (intervened_data_target.mean() - causal_object.data_new.mean()) ** 2
            )
            diff_value = objective_without_penalty - diff_target
        else:
            diff_value = objective_without_penalty
        
        # Compute transformation proportion
        trans_prop = 1 - (diff_value / causal_object.total)
        
        # Return results as list: [lambda, diff, trans_prop, x1, x2, ..., xn]
        return [lambda_val, diff_value, trans_prop] + result.x.tolist()
        
    except Exception as e:
        print(f"Error processing lambda={lambda_val:.4f}: {e}")
        return None


def get_attribution_scores_parallel_standalone(causal_object, de_cols=None, test='parametric', alpha=0.05, n_jobs=None):
    # Auto-detect CPUs from SLURM or fallback to local cores
    if n_jobs is None:
        n_jobs = int(os.environ.get("NCPUS", os.cpu_count()))
    
    if not de_cols:
        de_cols = causal_object.DE_naive(test, alpha)
    print('# of DE variables:', len(de_cols))

    def helper(col):
        return causal_object.get_distribution_change(col)

    # Run in parallel
    scores_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(helper)(col) for col in de_cols
    )
        
    attrib_scores = {}
    for col in causal_object.data_old.columns:
        attrib_scores[col] = 0
    for scores in scores_list:
        for k, v in scores.items():
            attrib_scores[k] += abs(v)
    
    sorted_attributions = sort_nodes(attrib_scores)
    
    return sorted_attributions


def get_perturbation_targets_by_specified_combs_parallel_standalone(
    causal_object,
    combs,
    minval: float = 0,
    maxval: float = 10000,
    maxiter: int = 10000,
    minimize_target_state_change: bool = False,
    gamma: float = 1e-2,
    nodes_need_constancy: Optional[List[Any]] = None,
    thres_change: float = 0.5,
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    verbose: int = 1
) -> pd.DataFrame:
    # Auto-detect CPUs (PBS-aware)
    if n_jobs is None:
        n_jobs = int(os.environ.get("NCPUS", os.cpu_count()))
    
    print(f"Using {n_jobs} workers for parallel optimization")
    
    # Pre-compute shared data to avoid serialization overhead
    shared_params = {
        'minval': minval,
        'maxval': maxval,
        'maxiter': maxiter,
        'minimize_target_state_change': minimize_target_state_change,
        'gamma': gamma,
        'nodes_need_constancy': nodes_need_constancy,
        'thres_change': thres_change
    }
    
    # Run parallel optimization
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_process_single_combination)(
            causal_object, comb, shared_params
        )
        for comb in combs
    )
    
    # Filter and format results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("Warning: No valid results found!")
        return pd.DataFrame(columns=['intervention targets', 'intervention values', 'Diff', 'TransProp'])
    
    print(f"Successfully optimized {len(valid_results)}/{len(combs)} combinations")
    
    # Create output DataFrame
    outputs = pd.DataFrame(
        valid_results, 
        columns=['intervention targets', 'intervention values', 'Diff']
    )
    outputs['TransProp'] = 1 - (outputs['Diff'] / causal_object.total)
    outputs['# of intervention targets'] = outputs['intervention targets'].map(lambda x: len(x))
    
    return outputs.sort_values(by='Diff', ignore_index=True)