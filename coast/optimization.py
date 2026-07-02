import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm
import itertools
from scipy.optimize import minimize
from scipy.stats import norm, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from dowhy.graph import is_root_node, get_ordered_predecessors, get_descendants
from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool
from joblib import Parallel, delayed
import os
import warnings

from coast.causal_model import *
from coast.distribution_change_new import *
from coast.utils import *



class CausalClass:
    def __init__(self, data_old, data_new, causal_graph, causal_model, causal_graph_new=None, causal_model_new=None):
        """
        initialize the class object
        old: source state
        new: target state
        """
        assert list(data_old.columns) == list(data_new.columns)
        # TODO: assert causal_graph and causal_model are consistent with data
        self.data_old = data_old
        self.data_new = data_new
        self.causal_graph = causal_graph
        self.causal_model = causal_model
        self.causal_graph_new = causal_graph_new
        self.causal_model_new = causal_model_new
        self.total = None

    
    def get_total_changes(self):
        """
        get the sum of the squares of variables between the source and target state data
        """
        changes = 0
        for node in self.causal_graph.nodes():
            changes += (self.data_old[node].mean()-self.data_new[node].mean())**2
        print('Total changes:', changes)
        self.total = changes

    
    def fit_old_model_on_new_data(self):
        """
        Fit the causal model entailed by the causal graph derived from the source state data
        On the target state data
        """
        causal_graph_copy = nx.DiGraph(self.causal_graph)
        causal_model_copy = gcm.StructuralCausalModel(causal_graph_copy)
        #gcm.auto.assign_causal_mechanisms(causal_model, data)
        custom_mechanisms(causal_graph_copy, causal_model_copy)
        gcm.fit(causal_model_copy, self.data_new)
        self.causal_model_on_new_data = causal_model_copy

    
    def check_mechanism_changes(self):
        """
        check if the causal mechanisms change for each node
        refit the causal models on the nodes whose mechanisms are not changed by using both the old and new data
        :return: a dictionary containing True (changed) or False (not changed) for each node
        """
        mechanism_changes = models_accounting_for_mechanism_change(
            self.causal_model,
            self.causal_model_on_new_data,
            self.data_old,
            self.data_new,
        )
        return mechanism_changes


    def DE_naive(self, test='parametric', alpha=0.05, correction_method='fdr_bh'):
        """
        Naive differential expression analysis
        """
        p_raw = []
        for col in self.data_old.columns:
            if test=='parametric':
                _, p_val = ttest_ind(self.data_old[col], self.data_new[col])
            elif test=='nonparametric':
                _, p_val = mannwhitneyu(self.data_old[col], self.data_new[col])
            p_raw.append(p_val)
        multi = multipletests(p_raw, alpha=alpha, method=correction_method)
        flags = multi[0]
        de_cols = list(self.data_old.columns[flags])
        return de_cols

    
    ################################################################################
    # Get accumulative distribution change attribution scores
    ################################################################################
    def get_distribution_change(self, col):
        print('Variable is:', col)
        scores = gcm.distribution_change_of_graphs(self.causal_model, self.causal_model_on_new_data, col, difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x))
        return scores
    
    def get_attribution_scores(self, de_cols=None, test='parametric', alpha=0.05):
        if not de_cols:
            de_cols = self.DE_naive(test, alpha)
        print('# of DE variables:', len(de_cols))
        
        attrib_scores = {}
        for col in self.data_old.columns:
            attrib_scores[col] = 0
        for i in tqdm(range(len(de_cols))):
            scores = self.get_distribution_change(de_cols[i])
            for k, v in scores.items():
                attrib_scores[k] += abs(v)
    
        sorted_attributions = sort_nodes(attrib_scores)
    
        return sorted_attributions
    
    def get_attribution_scores_parallel(self, de_cols=None, test='parametric', alpha=0.05):
        if not de_cols:
            de_cols = self.DE_naive(test, alpha)
        print('# of DE variables:', len(de_cols))

        pool = Pool()
        scores_list = list(tqdm(pool.imap(self.get_distribution_change, de_cols), total=len(de_cols)))
        pool.close() 
        pool.join()
        
        attrib_scores = {}
        for col in self.data_old.columns:
            attrib_scores[col] = 0
        for scores in scores_list:
            for k, v in scores.items():
                attrib_scores[k] += abs(v)
    
        sorted_attributions = sort_nodes(attrib_scores)
    
        return sorted_attributions


    def get_subgraph(self, nodes_to_intervene):
        """
        get the subgraph containing the descendants of intervened nodes (including the intervened nodes)
        """
        nodes_to_intervene = list(nodes_to_intervene)
        des_set = get_descendants(self.causal_graph, nodes_to_intervene)
        for item in nodes_to_intervene:
            des_set.add(item)
        sub_graph = self.causal_graph.subgraph(list(des_set))
        return sub_graph


    def get_subgraph_new(self, nodes_to_intervene):
        """
        get the subgraph containing the descendants of intervened nodes (including the intervened nodes) for the target state graph
        """
        nodes_to_intervene = list(nodes_to_intervene)
        des_set = get_descendants(self.causal_graph_new, nodes_to_intervene)
        for item in nodes_to_intervene:
            des_set.add(item)
        sub_graph = self.causal_graph_new.subgraph(list(des_set))
        return sub_graph


    def get_intervened_data(self, b, nodes_to_intervene, sub_graph):
        """
        :param b: a list of intervention values
        :param nodes_to_intervene: a list of nodes to intervene (matched to b)
        :param sub_graph: the subgraph derived from nodes_to_intervene
        :return: the dataframe after interventions
        """
        ordered_nodes = list(nx.topological_sort(sub_graph))
        mapped_d = {}
        for i, node in enumerate(nodes_to_intervene):
            mapped_d[node] = i
    
        input = self.data_old.copy()
        changes = 0
        for node in ordered_nodes:
            if is_root_node(sub_graph, node)==False:
                ordered_parents = get_ordered_predecessors(self.causal_graph, node)
                m = self.causal_model.causal_mechanism(node).prediction_model.sklearn_model
                input[node] = m.predict(input[ordered_parents])
            if node in nodes_to_intervene:
                input[node] += b[mapped_d[node]]
                
        return input


    def get_intervened_data_target(self, b, nodes_to_intervene, sub_graph):
        """
        :param b: a list of intervention values
        :param nodes_to_intervene: a list of nodes to intervene (matched to b)
        :param sub_graph: the subgraph derived from nodes_to_intervene
        :return: the dataframe of the target state after interventions
        """
        ordered_nodes = list(nx.topological_sort(sub_graph))
        mapped_d = {}
        for i, node in enumerate(nodes_to_intervene):
            mapped_d[node] = i
    
        input = self.data_new.copy()
        changes = 0
        for node in ordered_nodes:
            if is_root_node(sub_graph, node)==False:
                ordered_parents = get_ordered_predecessors(self.causal_graph_new, node)
                m = self.causal_model_new.causal_mechanism(node).prediction_model.sklearn_model
                input[node] = m.predict(input[ordered_parents])
            if node in nodes_to_intervene:
                input[node] += b[mapped_d[node]]
                
        return input

    
    def objective_function_comb(self, b, nodes_to_intervene, minimize_target_state_change=False, gamma=1e-2):
        """
        define the objective function to minimize using the combinatorial approach
        """
        # Only need to consider the changes on nodes_to_intervene and their descendants
        sub_graph = self.get_subgraph(nodes_to_intervene)
        intervened_data = self.get_intervened_data(b, nodes_to_intervene, sub_graph)

        changes = 0
        for node in self.causal_graph.nodes():
            changes += (intervened_data[node].mean()-self.data_new[node].mean())**2

        if minimize_target_state_change:
            sub_graph_new = self.get_subgraph_new(nodes_to_intervene)
            intervened_data_target = self.get_intervened_data_target(b, nodes_to_intervene, sub_graph_new)
            for node in self.causal_graph.nodes():
                changes += gamma*(intervened_data_target[node].mean()-self.data_new[node].mean())**2
    
        return changes

    def objective_function_reg(self, b, nodes_to_intervene, Lambda, weights, minimize_target_state_change=False, gamma=1e-2):
        """
        define the objective function to minimize using the regularization approach
        """
        # Only need to consider the changes on nodes_to_intervene and their descendants
        sub_graph = self.get_subgraph(nodes_to_intervene)
        intervened_data = self.get_intervened_data(b, nodes_to_intervene, sub_graph)

        changes = 0
        for node in self.causal_graph.nodes():
            changes += (intervened_data[node].mean()-self.data_new[node].mean())**2

        penalty = Lambda * np.dot(weights, abs(b))
        changes += penalty

        if minimize_target_state_change:
            sub_graph_new = self.get_subgraph_new(nodes_to_intervene)
            intervened_data_target = self.get_intervened_data_target(b, nodes_to_intervene, sub_graph_new)
            for node in self.causal_graph.nodes():
                changes += gamma*(intervened_data_target[node].mean()-self.data_new[node].mean())**2
    
        return changes

    
    def define_constraints(self, b, nodes_to_intervene, minval=0, maxval=10000, nodes_need_constancy=None, thres_change=0.5):
        """
        return an array of inequality constraints written as expressions that should be greater than 0
        """
        # Only need to consider the changes on nodes_to_intervene and their descendants
        sub_graph = self.get_subgraph(nodes_to_intervene)
        intervened_data = self.get_intervened_data(b, nodes_to_intervene, sub_graph)
        
        cons = []
        for node in sub_graph.nodes():
            cons.append(intervened_data[node].mean() - minval)
            cons.append(maxval - intervened_data[node].mean())

        if nodes_need_constancy:
            for node in nodes_need_constancy:
                cons.append(thres_change + self.data_old[node].mean() - intervened_data[node].mean())
                cons.append(thres_change + intervened_data[node].mean() - self.data_old[node].mean())
    
        return np.array(cons)


    def define_bounds(self, nodes_to_intervene, bounds_type=None):
        """
        Define bounds for the optimization variables.
        :param nodes_to_intervene: list of nodes being optimized
        :param bounds_type: None (no bounds), "positive" (all >= 0), or "negative" (all <= 0)
        :return: list of (min, max) tuples or None
        """
        if bounds_type is None:
            return None
        elif bounds_type == "positive":
            return [(0, None)] * len(nodes_to_intervene)
        elif bounds_type == "negative":
            return [(None, 0)] * len(nodes_to_intervene)
        else:
            raise ValueError(f"Invalid bounds_type: {bounds_type}. Must be None, 'positive', or 'negative'.")


    def get_candidate_nodes(self, sorted_attributions, search_range=100, nodes_to_include=None, nodes_to_remove=None):
        """
        get the nodes with top attribution scores
        include or remove certain nodes if given
        """
        top_nodes = list(sorted_attributions.keys())[:search_range]
        top_nodes = set(top_nodes)
        if nodes_to_include:
            for node in nodes_to_include:
                if node not in top_nodes:
                    top_nodes.add(node)
        if nodes_to_remove:
            for node in nodes_to_remove:
                top_nodes.discard(node)
        return list(top_nodes)

    def adaptive_selection(self, sorted_attributions, delta=0.01):
        """
        Select variables with top attribution scores using adaptive stopping rule:
        Stop when next variable's contribution is negligible compared to cumulative contribution.
    
        Parameters:
            delta (float): threshold ratio (e.g., 0.01 for 1%)
    
        Returns:
            selected_vars (list): top variables selected
            normalized_scores (dict): scores normalized to sum to 100
        """
        total = sum(sorted_attributions.values())
        normalized_scores = {k: (v / total) * 100 for k, v in sorted_attributions.items()}
        sorted_items = list(normalized_scores.items())
    
        selected_vars = [sorted_items[0][0]]  # start with top variable
        cumulative = sorted_items[0][1] 
        for i in range(1, len(sorted_items)):
            next_score = sorted_items[i][1]
            if next_score / cumulative < delta:
                break
            selected_vars.append(sorted_items[i][0])
            cumulative += next_score
    
        return selected_vars, normalized_scores

    
    ################################################################################
    # Combinatorial approach
    ################################################################################

    def optimize_by_comb(self, params):
        # function for parallel computing (within CausalClass)
        comb, minval, maxval, maxiter, minimize_target_state_change, gamma, nodes_need_constancy, thres_change, bounds_type = params
        print(comb)
        try:
            f = lambda x: self.objective_function_comb(x, nodes_to_intervene=comb, minimize_target_state_change=minimize_target_state_change, gamma=gamma)
            b0 = [0.]*len(comb)
            ineq_cons = {'type': 'ineq',
                         'fun' : lambda x: self.define_constraints(x, nodes_to_intervene=comb, minval=minval, maxval=maxval, nodes_need_constancy=nodes_need_constancy, thres_change=thres_change)}
            bounds = self.define_bounds(comb, bounds_type=bounds_type)
            res = minimize(f, b0, constraints=ineq_cons, bounds=bounds, options={'maxiter': maxiter})
            if minimize_target_state_change:
                sub_graph_new = self.get_subgraph_new(comb)
                intervened_data_target = self.get_intervened_data_target(res.x, comb, sub_graph_new)
                diff_target = gamma * sum((intervened_data_target.mean()-self.data_new.mean())**2)
                return (comb, res.x, res.fun - diff_target)
            else:
                return (comb, res.x, res.fun)
        except:
            print('Error...')
            return None

    def get_perturbation_targets_by_comb(self, top_nodes, attrib_scores, num_interventions=3, max_combinations=500, minval=0, maxval=10000, maxiter=10000, minimize_target_state_change=False, gamma=1e-2, nodes_need_constancy=None, thres_change=0.5, bounds_type=None):
        solutions = []
        diff_values = []
        valid_combs = []

        # get all the combinations containing num_interventions elements
        combs = [item for item in itertools.combinations(top_nodes, num_interventions)]

        # sort the combinations by the sum of the attribution scores of the elements and keep the top max_combinations combinations
        scores = {}
        for comb in combs:
            score = 0
            for node in comb:
                score += attrib_scores[node]
            scores[comb] = score
        sorted_scores = sort_nodes(scores)
        top_combs = list(sorted_scores.keys())[:max_combinations]
        
        for i in tqdm(range(len(top_combs))):
            comb = top_combs[i]
            print(comb)
            try:
                f = lambda x: self.objective_function_comb(x, nodes_to_intervene=comb, minimize_target_state_change=minimize_target_state_change, gamma=gamma)
                b0 = [0.]*len(comb)
                ineq_cons = {'type': 'ineq',
                             'fun' : lambda x: self.define_constraints(x, nodes_to_intervene=comb, minval=minval, maxval=maxval, nodes_need_constancy=nodes_need_constancy, thres_change=thres_change)}
                bounds = self.define_bounds(comb, bounds_type=bounds_type)
                res = minimize(f, b0, constraints=ineq_cons, bounds=bounds, options={'maxiter': maxiter})
                valid_combs.append(comb)
                solutions.append(res.x)
                if minimize_target_state_change:
                    sub_graph_new = self.get_subgraph_new(comb)
                    intervened_data_target = self.get_intervened_data_target(res.x, comb, sub_graph_new)
                    diff_target = gamma * sum((intervened_data_target.mean()-self.data_new.mean())**2)
                    diff_values.append(res.fun - diff_target)
                else:
                    diff_values.append(res.fun)
            except:
                print('Error...')

        outputs = pd.DataFrame(zip(valid_combs, solutions, diff_values), columns=['intervention targets','intervention values','Diff'])
        outputs['TransProp'] = 1 - (outputs['Diff']/self.total)
        outputs = outputs.sort_values(by='Diff', ignore_index=True)
        return outputs
    
    def get_perturbation_targets_by_comb_parallel(self, top_nodes, attrib_scores, num_interventions=3, max_combinations=500, minval=0, maxval=10000, maxiter=10000, minimize_target_state_change=False, gamma=1e-2, nodes_need_constancy=None, thres_change=0.5, bounds_type=None):
        solutions = []
        diff_values = []
        valid_combs = []

        # get all the combinations containing num_interventions elements
        combs = [item for item in itertools.combinations(top_nodes, num_interventions)]

        # sort the combinations by the sum of the attribution scores of the elements and keep the top max_combinations combinations
        scores = {}
        for comb in combs:
            score = 0
            for node in comb:
                score += attrib_scores[node]
            scores[comb] = score
        sorted_scores = sort_nodes(scores)
        top_combs = list(sorted_scores.keys())[:max_combinations]

        # parallel computing
        params_list = []
        for comb in top_combs:
            params_list.append((comb, minval, maxval, maxiter, minimize_target_state_change, gamma, nodes_need_constancy, thres_change, bounds_type))
        pool = Pool()
        res_list = list(tqdm(pool.imap(self.optimize_by_comb, params_list, chunksize=10), total=len(top_combs)))
        pool.close() 
        pool.join()

        res_list = [res for res in res_list if res is not None]
        for res in res_list:
            valid_combs.append(res[0])
            solutions.append(res[1])
            diff_values.append(res[2])

        outputs = pd.DataFrame(zip(valid_combs, solutions, diff_values), columns=['intervention targets','intervention values','Diff'])
        outputs['TransProp'] = 1 - (outputs['Diff']/self.total)
        outputs = outputs.sort_values(by='Diff', ignore_index=True)
        return outputs


    ################################################################################
    # Regularization approach
    ################################################################################

    def compute_lambda_range(self, weights=None, epsilon=1e-3, num_points=20):
        """
        Compute lambda_max using KKT conditions and generate a logarithmic grid of lambda values.

        Parameters:
            weights: L1 penalty weights w_i (normalized weights based on attribution scores). If None, equal weights are used.
            epsilon: Fraction for lambda_min (default=1e-3)
            num_points: Number of lambda values in the grid (default=20)

        Returns:
            lambda_max (float): Upper bound for lambda
            lambda_grid (ndarray): Logarithmic grid of lambda values
        """
        # Compute column means for source and target
        source_mean = self.data_old.mean(axis=0).values
        target_mean = self.data_new.mean(axis=0).values

        # If weights are not provided, assign equal weights
        if weights is None:
            weights = np.ones_like(source_mean)

        # Gradient at alpha = 0 (assuming additive shifts and identity Jacobian)
        gradient = 2 * (source_mean - target_mean) 

        # Compute lambda_max using KKT condition: |g_i| <= lambda * w_i
        lambda_max = max(np.abs(gradient)) / np.min(weights)

        # Compute lambda_min
        lambda_min = epsilon * lambda_max

        # Generate logarithmic grid
        lambda_grid = lambda_max * (lambda_min / lambda_max) ** (np.arange(num_points) / (num_points - 1))

        return lambda_max, lambda_grid

    
    def optimize_by_reg(self, params):
        # function for parallel computing (within CausalClass)
        log, top_nodes, minval, maxval, maxiter, weights, minimize_target_state_change, gamma, nodes_need_constancy, thres_change, bounds_type = params
        Lambda = 10**log
        print(log)
        try:
            f = lambda x: self.objective_function_reg(x, nodes_to_intervene=top_nodes, Lambda=Lambda, weights=weights, minimize_target_state_change=minimize_target_state_change, gamma=gamma)
            b0 = [0.]*len(top_nodes)
            ineq_cons = {'type': 'ineq',
                         'fun' : lambda x: self.define_constraints(x, nodes_to_intervene=top_nodes, minval=minval, maxval=maxval, nodes_need_constancy=nodes_need_constancy, thres_change=thres_change)}
            bounds = self.define_bounds(top_nodes, bounds_type=bounds_type)
            res = minimize(f, b0, constraints=ineq_cons, bounds=bounds, options={'maxiter': maxiter})
            pnl = Lambda * np.dot(weights, abs(res.x))
            min_vals_wout_penalty = res.fun - pnl
            if minimize_target_state_change:
                sub_graph_new = self.get_subgraph_new(top_nodes)
                intervened_data_target = self.get_intervened_data_target(res.x, top_nodes, sub_graph_new)
                diff_target = gamma * sum((intervened_data_target.mean()-self.data_new.mean())**2)
                return (log, res.x, min_vals_wout_penalty - diff_target)
            else:
                return (log, res.x, min_vals_wout_penalty)
        except:
            print('Error...')
            return None

    def get_perturbation_targets_by_reg(self, top_nodes, attrib_scores, min_log_lambda=-2, max_log_lambda=4, step_size=0.5, minval=0, maxval=10000, maxiter=10000, minimize_target_state_change=False, gamma=1e-2, nodes_need_constancy=None, thres_change=0.5, bounds_type=None):
        solutions = {}
        diff_values = {}

        # assign weights to each of the top_nodes by 1/attribution score and normalize the weights
        weights = []
        min_score = 1e-2
        for node in top_nodes:
            if attrib_scores[node] > min_score:
                weights.append(1/attrib_scores[node])
            else:
                weights.append(1/min_score)
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        logs = np.arange(min_log_lambda, max_log_lambda+step_size, step_size)
        for i in tqdm(range(len(logs))):
            log = logs[i]
            Lambda = 10**log
            print(log)
            try:
                f = lambda x: self.objective_function_reg(x, nodes_to_intervene=top_nodes, Lambda=Lambda, weights=weights, minimize_target_state_change=minimize_target_state_change, gamma=gamma)
                b0 = [0.]*len(top_nodes)
                ineq_cons = {'type': 'ineq',
                             'fun' : lambda x: self.define_constraints(x, nodes_to_intervene=top_nodes, minval=minval, maxval=maxval, nodes_need_constancy=nodes_need_constancy, thres_change=thres_change)}
                bounds = self.define_bounds(top_nodes, bounds_type=bounds_type)
                res = minimize(f, b0, constraints=ineq_cons, bounds=bounds, options={'maxiter': maxiter})
                solutions[log] = res.x
                pnl = Lambda * np.dot(weights, abs(res.x))
                min_vals_wout_penalty = res.fun - pnl
                diff_values[log] = min_vals_wout_penalty
                if minimize_target_state_change:
                    sub_graph_new = self.get_subgraph_new(top_nodes)
                    intervened_data_target = self.get_intervened_data_target(res.x, top_nodes, sub_graph_new)
                    diff_target = gamma * sum((intervened_data_target.mean()-self.data_new.mean())**2)
                    diff_values[log] = min_vals_wout_penalty - diff_target
            except:
                print('Error...')

        outputs = []
        for log in list(solutions.keys()):
            output = [log] + [diff_values[log]] + [1 - (diff_values[log]/self.total)] + solutions[log].tolist()
            outputs.append(output)
        outputs = pd.DataFrame(outputs)
        outputs.columns = ['Log10Lambda', 'Diff', 'TransProp'] + top_nodes

        outputs = outputs.sort_values(by='Diff', ignore_index=True)
        return outputs
    
    def get_perturbation_targets_by_reg_parallel(self, top_nodes, attrib_scores, min_log_lambda=-2, max_log_lambda=4, step_size=0.5, minval=0, maxval=10000, maxiter=10000, minimize_target_state_change=False, gamma=1e-2, nodes_need_constancy=None, thres_change=0.5, bounds_type=None):
        solutions = {}
        diff_values = {}

        # assign weights to each of the top_nodes by 1/attribution score and normalize the weights
        logs = np.arange(min_log_lambda, max_log_lambda+step_size, step_size)
        weights = []
        min_score = 1e-2
        for node in top_nodes:
            if attrib_scores[node] > min_score:
                weights.append(1/attrib_scores[node])
            else:
                weights.append(1/min_score)
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # parallel computing
        params_list = []
        for log in logs:
            params_list.append((log, top_nodes, minval, maxval, maxiter, weights, minimize_target_state_change, gamma, nodes_need_constancy, thres_change, bounds_type))
        pool = Pool()
        res_list = list(tqdm(pool.imap(self.optimize_by_reg, params_list, chunksize=1), total=len(logs)))
        pool.close() 
        pool.join()

        res_list = [res for res in res_list if res is not None]

        outputs = []
        for res in res_list:
            output = [res[0]] + [res[2]] + [1 - (res[2]/self.total)] + res[1].tolist()
            outputs.append(output)
        outputs = pd.DataFrame(outputs)
        outputs.columns = ['Log10Lambda', 'Diff', 'TransProp'] + top_nodes

        outputs = outputs.sort_values(by='Diff', ignore_index=True)
        return outputs


    def process_results_by_reg(self, outputs, thres_val=1e-1):
        """
        process the results of regularization approach:
        count the number of intervention targets, and record their names and shift values
        """
        top_nodes = outputs.columns[3:]
        
        number_of_intervention_targets = []
        intervention_targets = []
        intervention_values = []
        for i in range(outputs.shape[0]):
            v = outputs.iloc[i, 3:].values
            idx = np.where(abs(v) > thres_val)[0]
            number = len(idx)
            targets = top_nodes[idx]
            values = v[idx]
            number_of_intervention_targets.append(number)
            intervention_targets.append(targets)
            intervention_values.append(values)

        processed_results = outputs.iloc[:,:3].copy()
        processed_results['# of intervention targets'] = number_of_intervention_targets
        processed_results['intervention targets'] = intervention_targets
        processed_results['intervention values'] = intervention_values
        
        return processed_results
    

    def get_affected_nodes(self, intervened_nodes, intervened_values, test='parametric', alpha=0.05, thres_logFC=2):
        """
        get the nodes that change the most on the source state data after the given intervention by a simple DE analysis
        :param alpha: significance level
        :param thres_logFC: threshold for absolute logFC
        :return: a dictionary containing the nodes with significant change, and their corresponding p-values and logFCs
        """
        ### By using dowhy functions
        #intervention_mechanism = {}
        #for i, node in enumerate(intervened_nodes):
        #    intervention_mechanism[node] = lambda x:x+intervened_values[i]
        #intervened_data = gcm.interventional_samples(self.causal_model, intervention_mechanism, self.data_old)
        
        sub_graph = self.get_subgraph(intervened_nodes)
        intervened_data = self.get_intervened_data(intervened_values, intervened_nodes, sub_graph)

        p_raw = []
        logFC = []
        for col in self.data_old.columns:
            if test=='parametric':
                _, p_val = ttest_ind(self.data_old[col], intervened_data[col])
            elif test=='nonparametric':
                _, p_val = mannwhitneyu(self.data_old[col], intervened_data[col])
            p_raw.append(p_val)
            logFC.append(intervened_data[col].mean() - self.data_old[col].mean())
        multi = multipletests(p_raw, alpha=alpha, method='fdr_bh')
        p_adj = multi[1]

        post_changes_df = pd.DataFrame(zip(self.data_old.columns, p_adj, logFC), columns=['Genes', 'Adj P-val', 'logFC'])
        post_changes_df = post_changes_df[(post_changes_df['Adj P-val']<alpha) & (abs(post_changes_df['logFC'])>thres_logFC)]
        post_changes_df = post_changes_df.sort_values(by='Adj P-val', ignore_index=True)

        return post_changes_df


    def get_pathway_changes(self, intervened_nodes, post_changes):
        """
        get the pathways that change the most after the given intervention
        paths start from the root nodes of the subgraph of the intervened nodes and end at the leaf nodes
        the change of a path is calculated as the sum of the absolute changes of all the nodes on the path
        :return: a dictionary containing all the paths and their corresponding changes
        """
        sub_graph = self.get_subgraph(intervened_nodes)
    
        roots = [v for v in intervened_nodes if is_root_node(sub_graph, v)] # roots = (v for v, d in G.in_degree() if d == 0)
        leaves = [v for v, d in sub_graph.out_degree() if d == 0]
        all_paths = []
        for root in roots:
            paths = nx.all_simple_paths(sub_graph, root, leaves)
            all_paths.extend(paths)

        pathway_changes = {}
        for path in all_paths:
            changes = 0
            for node in path:
                changes += abs(post_changes[node])
            path_key = '->'.join(path)
            pathway_changes[path_key] = changes

        return pathway_changes