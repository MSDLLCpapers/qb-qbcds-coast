import numpy as np
from scipy import stats

def bh_correction(pvals):
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    adjusted = np.zeros(n)
    adjusted[sorted_idx[-1]] = sorted_pvals[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i + 1]], sorted_pvals[i] * n / (i + 1))
    return np.minimum(adjusted, 1.0)

def run_ttest(x_file, y_file, gene_file, out_file, label):
    X = np.loadtxt(x_file)
    Y = np.loadtxt(y_file)
    gene_names = np.loadtxt(gene_file, dtype=str)

    mo_idx = Y[:, 0] == 1
    opc_idx = Y[:, 1] == 1
    X = X.T
    X_mo = X[mo_idx, :]
    X_opc = X[opc_idx, :]
    print(f"\n=== {label} ===")
    print(f"MO cells: {X_mo.shape[0]}, OPC cells: {X_opc.shape[0]}, Genes: {X.shape[1]}")

    n_genes = X.shape[1]
    t_stats = np.zeros(n_genes)
    p_values = np.zeros(n_genes)
    log_fc = np.zeros(n_genes)

    for i in range(n_genes):
        t_stats[i], p_values[i] = stats.ttest_ind(X_mo[:, i], X_opc[:, i], equal_var=False)
        log_fc[i] = X_mo[:, i].mean() - X_opc[:, i].mean()

    p_bh = bh_correction(p_values)

    alpha = 0.05
    print(f"At alpha = {alpha}: {np.sum(p_bh < alpha)} significant genes (BH-corrected)")

    sorted_idx = np.argsort(p_bh)
    with open(out_file, 'w') as f:
        f.write('gene,log_fc,t_stat,p_value,p_bh\n')
        for i in sorted_idx:
            f.write(f'{gene_names[i]},{log_fc[i]:.6f},{t_stats[i]:.6f},{p_values[i]:.6e},{p_bh[i]:.6e}\n')

    print(f"\nTop 10 DE genes (by BH-adjusted p-value):")
    print(f"{'Gene':>15s} {'logFC':>8s} {'p_BH':>12s}")
    for i in sorted_idx[:10]:
        print(f"{gene_names[i]:>15s} {log_fc[i]:>8.3f} {p_bh[i]:>12.2e}")
    print(f"Results saved to {out_file}")

# Global >30% filter
run_ttest('chen_X.txt', 'chen_Y.txt', 'gene_names.txt', 'de_results_ttest_global.csv', 'Global >30% filter')

# Per-group >10% filter
run_ttest('chen_X2.txt', 'chen_Y2.txt', 'gene_names2.txt', 'de_results_ttest_pergroup.csv', 'Per-group >10% filter')
