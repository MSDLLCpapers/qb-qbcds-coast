import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from rpy2.robjects import conversion, default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import time


def load_expression_data(expression_file, index_col=None, transpose=False):
    # Read the CSV file
    df = pd.read_csv(expression_file, index_col=index_col)
    # Transpose the DataFrame
    if transpose:
        df = df.transpose()
    return df

def load_target_genes(target_file):
    return pd.read_csv(target_file, header=None).squeeze().tolist()

def run_genie3(expression_data, target_genes=None, n_cores=4, k=7, n_trees=100):
    genie3 = importr('GENIE3')
    with localconverter(default_converter + pandas2ri.converter):
        r_df = conversion.py2rpy(expression_data)
    r_matrix = robjects.r['as.matrix'](r_df)
    if target_genes:
        weight_matrix = genie3.GENIE3(r_matrix, nCores=n_cores, targets=target_genes, treeMethod="ET", K=k, nTrees=n_trees)
    else:
        weight_matrix = genie3.GENIE3(r_matrix, nCores=n_cores, treeMethod="ET", K=k, nTrees=n_trees)
    link_list = genie3.getLinkList(weight_matrix)
    with localconverter(default_converter + pandas2ri.converter):
        link_list_df = conversion.rpy2py(link_list)
    link_list_df.columns = ['Regulator', 'Target', 'Weight']  # Fix column names
    return weight_matrix, link_list_df

def first_level_filtering(link_list_df, top_n1):
    return (
        link_list_df.sort_values(['Target', 'Weight'], ascending=[True, False])
        .groupby('Target', observed=True)  # Fix FutureWarning
        .head(top_n1)
        .reset_index(drop=True)
    )

def create_causal_dataset(expression_data, regulators, targets, output_file):
    # output is of format samples * genes
    causal_genes = set(regulators).union(set(targets))
    causal_dataset = expression_data.loc[list(causal_genes)].transpose()
    causal_dataset.to_csv(output_file, index=False, sep=',')

def compute_hybrid_scores(weight_matrix, gene_names, target_genes, alpha=0.7):
    n_genes = len(gene_names)
    genie3_matrix = np.array(weight_matrix).reshape((n_genes, n_genes))

    G = nx.DiGraph()
    for i, regulator in enumerate(gene_names):
        for j, target in enumerate(gene_names):
            weight = genie3_matrix[i, j]
            if weight > 0:
                G.add_edge(regulator, target, weight=weight)

    if G.number_of_nodes()>100:
        betweenness = nx.betweenness_centrality(G, k=100, weight='weight', normalized=True)
    else:
        betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)

    scaler = MinMaxScaler()
    normalized_weights = scaler.fit_transform(genie3_matrix)
    betweenness_array = np.array([betweenness[gene] for gene in gene_names]).reshape(-1, 1)
    normalized_betweenness = scaler.fit_transform(betweenness_array).flatten()

    def compute_scores_for_target(target):
        target_idx = gene_names.index(target)
        scores = []
        for regulator_idx, regulator in enumerate(gene_names):
            if regulator != target:
                weight = normalized_weights[regulator_idx, target_idx]
                centrality = normalized_betweenness[regulator_idx]
                hybrid_score = alpha * weight + (1-alpha) * centrality
                scores.append({
                    'Target': target,
                    'Regulator': regulator,
                    'Weight': weight,
                    'Betweenness': centrality,
                    'Hybrid Score': hybrid_score
                })
        return scores

    results = Parallel(n_jobs=-1)(delayed(compute_scores_for_target)(target) for target in target_genes)
    hybrid_scores = [score for sublist in results for score in sublist]
    return pd.DataFrame(hybrid_scores)

def run_pipeline(expression_data, target_genes, top_n1=10, top_n2=5, n_cores=4, k=7, n_trees=100, run_second_filtering=True, alpha=0.7):
    start = time.time()
    # expression_data is of format genes * samples
    #expression_data = load_expression_data(expression_file)
    #target_genes = load_target_genes(target_file)

    weight_matrix, link_list_df = run_genie3(expression_data, target_genes, n_cores, k, n_trees)
    top_n1_df = first_level_filtering(link_list_df, top_n1)
    create_causal_dataset(expression_data, top_n1_df['Regulator'], top_n1_df['Target'], "Causal_Discovery_Dataset_1st_Level_Filtering.csv")

    if run_second_filtering:
        unique_regulators = set(top_n1_df['Regulator'].unique())
        unique_targets = set(top_n1_df['Target'].unique())
        TSG = unique_regulators.union(unique_targets)
        second_dataset = expression_data.loc[list(TSG)]
        weight_matrix2, _ = run_genie3(second_dataset, None, n_cores, k, n_trees)
        hybrid_df = compute_hybrid_scores(weight_matrix2, list(TSG), target_genes, alpha)
        top_n2_df = hybrid_df.sort_values(by=['Target', 'Hybrid Score'], ascending=[True, False]).groupby('Target', observed=True).head(top_n2).reset_index(drop=True)
        create_causal_dataset(expression_data, top_n2_df['Regulator'], top_n2_df['Target'], "Causal_Discovery_Dataset_2nd_Level_Filtering.csv")

    print(f"Feature selection completed in {time.time() - start:.2f} seconds.")

#############################################################################################################################################################################################





#################################################################################
# Running Example
#################################################################################


######################################################
# Generate Synthetic Data for Testing
######################################################

# Generate Synthetic Data for GENIE3
# import numpy as np
# import pandas as pd

# Parameters
# num_genes = 10000
# num_samples = 500
# num_targets = 700

# Generate gene names
# genes = [f"Gene{i+1}" for i in range(num_genes)]

# Generate random expression data (normal distribution)
# expression_data = np.random.normal(loc=0, scale=1, size=(num_genes, num_samples))

# Create DataFrame with genes as index and samples as columns
# expr_df = pd.DataFrame(expression_data, index=genes, columns=[f"Sample{i+1}" for i in range(num_samples)])

# Save expression data as CSV (comma separated)
# transpose the dataframe to make rows:samples x columns:genes
# expr_df = expr_df.transpose()
# expr_df.to_csv("expression_data.csv", sep=",")

# Randomly select target genes
# target_genes = np.random.choice(genes, size=num_targets, replace=False)

# Save targets file
# with open("targets.txt", "w") as f:
#     for gene in target_genes:
#         f.write(gene + "\n")

# print("Files generated: expression_data.csv and targets.txt")



# ######################################################
# # Run the Feature Selection Pipeline
# ######################################################
# import time
# start = time.time()

# Example usage:

# run_pipeline('expression_data.csv', 'targets.txt', top_n1=10, top_n2=5, n_cores=30, run_second_filtering=True)

# end = time.time()
# print(f"Wall time: {end - start:.6f} seconds")
