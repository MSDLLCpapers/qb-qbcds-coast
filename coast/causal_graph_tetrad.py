#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
#import pydot
#import graphviz
#from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
#from causallearn.utils.GraphUtils import GraphUtils

import jpype.imports
import tools.translate as tr
import tools.TetradSearch as ts
import edu.cmu.tetrad.data as td
import edu.cmu.tetrad.graph as gr
import java.lang as lang
from edu.cmu.tetrad.util import Params, Parameters


def causal_discovery(data, path, custom_name, data_type=None, num_categories=None,
                     algorithm='fges', score=None, test=None, alpha=0.05, 
                     max_degree=10, parallelized=True, number_resampling=0, seed=-1, knowledge_file=None):
    """Run causal discovery algorithms on processed data to learn a causal graph structure.
    
    :param data_type: Data type of variables, 'continuous', 'mixed' or 'discrete'.
    :param num_categories: Maximum number of categories a discrete variable can have.
    :param algorithm: Which causal discovery algorithm to use, 'fges', 'lingam', 'pc' or 'fci'. See the github homepage or tutorial for more information on which one to choose.
    :param score: Which score to use, for score-based algorithms. Options are 'sem-bic', 'cg-bic', 'dg-bic' or 'bdeu'. See the github homepage or tutorial for more information on which one to choose.
    :param test: Which test to use, for test-based algorithms. Options are 'fisher-z', 'cg-lr', 'dg-lr', or 'chi-square'. See the github homepage or tutorial for more information on which one to choose.
    :param alpha: Cutoff for p value of the test, set to 0.05 by default.
    :param max_degree: Maximum degree of a node in the graph, set to 10 by default.
    :param parallelized: Whether the search is parallelized.
    :param number_resampling: Number of bootstrap resamples used for causal discovery. If set to 0, the algorithm will be run one time using all samples; if greater than 0, the algorithm will be run multiple times using bootstrap samples, and edge probabilities will be returned, which will provide information about the confidence of edge existence and direction. By default, number_resampling is set to 0.
    :param seed: Seed number for reproducibility.
    :param knowledge_file: File containing prior knowledge about causal relations between variables.
    :return: Save both CPDAG and DAG.
    """
    search = ts.TetradSearch(data)  # Make a TetradSearch instance to run searches against.
    search.set_verbose(False)

    if score=='dg-bic':
        search.use_degenerate_gaussian_score()
    elif score=='cg-bic':
        search.use_conditional_gaussian_score()
    elif score=='sem-bic':
        search.use_sem_bic()
    elif score=='bdeu':
        search.use_bdeu(sample_prior=10, structure_prior=0)

    if test=='fisher-z':
        search.use_fisher_z(alpha=alpha)
    elif test=='chi-square':
        search.use_chi_square(alpha=alpha)
    elif test=='dg-lr':
        search.use_degenerate_gaussian_test(alpha=alpha)
    elif test=='cg-lr':
        search.use_conditional_gaussian_test(alpha=alpha)

    if data_type:
        search.params.set(Params.DATA_TYPE, data_type)
    if num_categories:
        search.params.set(Params.NUM_CATEGORIES, num_categories)
    if max_degree:
        search.params.set(Params.MAX_DEGREE, max_degree)
    if parallelized:
        search.params.set(Params.PARALLELIZED, parallelized)

    if number_resampling>0:
        search.set_bootstrapping(numberResampling=number_resampling, percent_resample_size=90, add_original=True, with_replacement=False, resampling_ensemble=1, seed=seed)

    if knowledge_file:
        search.load_knowledge(knowledge_file)

    if algorithm=='fges':
        search.run_fges(max_degree=max_degree, parallelized=parallelized)
    elif algorithm=='pc':
        search.run_pc()
    elif algorithm=='fci':
        search.run_fci()
    elif algorithm=='lingam':
        search.run_ica_lingam()
    
    res = search.get_string()
    text_file = open(f'{path}/{algorithm}_{custom_name}.txt', 'w')
    text_file.write(str(res))
    text_file.close()

    if algorithm!='fci':
        dag_res = search.get_dag_string()
        text_file = open(f'{path}/{algorithm}_dag_{custom_name}.txt', 'w')
        text_file.write(str(dag_res))
        text_file.close()
        