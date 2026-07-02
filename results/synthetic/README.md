## Running COAST on synthetic datasets

- `run.py` contains the code to generate the synthetic data and run COAST, given the input parameters
n_nodes (number of nodes in the graph), edge_prob (edge probability of the graph), noise_level (1.0, 3.0, or 5.0, corresponding to "low", "medium", or "high"), noise_type ("gaussian" or "non-gaussian"), a_size (number of intervention targets), and save_dir (the directory to save results).

- `analysis.ipynb` contains the code to evaluate the results of COAST of a specific run, using evaluation metrics recall@k, transition percentage@k, and average transition percentage,
where k is the ground-truth intervention size.

- `results_summary.ipynb` contains the code to summarize the results of COAST across all experimental runs.
