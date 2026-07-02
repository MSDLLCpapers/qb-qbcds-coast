import random
from collections import defaultdict
from scipy.stats import norm, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import pandas as pd

def sort_nodes(values, decreasing=True):
    sorted_nodes = sorted(values.items(), key=lambda x:abs(x[1]), reverse=True)
    
    if decreasing==False:
        sorted_nodes = sorted_nodes[::-1]

    return dict(sorted_nodes)


def post_DE(source_data, target_data, test='parametric', alpha=0.05, correction_method='fdr_bh'):
    diff_mean = abs(source_data.mean() - target_data.mean())
    df = diff_mean.to_frame(name="diff_mean")
    p_raw = []
    for col in source_data.columns:
        if test=='parametric':
            _, p_val = ttest_ind(source_data[col], target_data[col])
        elif test=='nonparametric':
            _, p_val = mannwhitneyu(source_data[col], target_data[col])
        p_raw.append(p_val)
    multi = multipletests(p_raw, alpha=alpha, method=correction_method)
    df['p_raw'] = p_raw
    df['p_adj'] = multi[1]
    
    df = df.sort_values(by=['p_adj','p_raw'])
    df_sig = df[df['p_adj']<alpha]
    df_nonsig = df[df['p_adj']>=alpha]
    df_sig = df_sig.sort_values(by='diff_mean', ascending=False)
    df = pd.concat([df_sig, df_nonsig], axis=0)
    return df


def select_top_n_with_random_ties(sorted_dict, n):
    """
    Select top N keys from a sorted dictionary (sorted by values, decreasing order).
    When there are ties, randomly select from those tied keys.
    
    Args:
        sorted_dict: Dictionary sorted by values in decreasing order
        n: Number of keys to select
    
    Returns:
        List of selected keys
    """
    if n <= 0:
        return []
    
    if n >= len(sorted_dict):
        return list(sorted_dict.keys())
    
    # Group keys by their values
    value_groups = defaultdict(list)
    for key, value in sorted_dict.items():
        value_groups[value].append(key)
    
    # Get unique values in decreasing order
    unique_values = sorted(value_groups.keys(), reverse=True)
    
    selected_keys = []
    
    # Iterate through value groups in decreasing order
    for value in unique_values:
        keys_with_this_value = value_groups[value]
        remaining_slots = n - len(selected_keys)
        
        if remaining_slots <= 0:
            break
        
        if len(keys_with_this_value) <= remaining_slots:
            # All keys in this group can be added
            selected_keys.extend(keys_with_this_value)
        else:
            # Randomly sample from this group to fill remaining slots
            sampled_keys = random.sample(keys_with_this_value, remaining_slots)
            selected_keys.extend(sampled_keys)
            break
    return selected_keys
    

# Elbow point selection for top nodes

# Replace/Refine (estimated) attribution scores by single-node intervention results


def visualize_comb(solutions_comb, candidate_nodes, top_combinations=50):
    counts={}
    intv_vals={}
    for node in candidate_nodes:
        counts[node]=0
        intv_vals[node]=[]

    for i in range(top_combinations):
        for x in solutions_comb['intervention targets'][i]:
            counts[x]+=1
        for node in candidate_nodes:
            intv_vals[node].append(0)
        for j in range(len(solutions_comb['intervention targets'][i])):
            target_node = solutions_comb['intervention targets'][i][j]
            intv_vals[target_node][-1] = solutions_comb['intervention values'][i][j]
            
    freqs = {k:v/top_combinations for k,v in counts.items()}
    intv_vals = pd.DataFrame(intv_vals)
    return freqs, intv_vals

# Example Usage
#freqs, intv_vals = visualize_comb(solutions_comb, top_nodes)
#dowhy_plot.bar_plot(freqs, ylabel='Frequencies')
#sns.heatmap(intv_vals, cmap='coolwarm', center=0)


def visualize_reg(solutions_reg, solutions_reg_processed, remove_duplicates=False, thres_val=0.1):
    
    solutions_reg_nonzero = solutions_reg_processed[solutions_reg_processed['# of intervention targets']>0]
    solutions_reg_nonzero_deduplicated = solutions_reg_nonzero.drop_duplicates(subset=['# of intervention targets'])
    
    solutions_reg_reduced1 = solutions_reg.loc[solutions_reg_nonzero_deduplicated.index]
    solutions_reg_reduced1 = solutions_reg_reduced1.reset_index(drop=True)

    solutions_reg_reduced2 = solutions_reg.loc[solutions_reg_nonzero.index]
    solutions_reg_reduced2 = solutions_reg_reduced2.reset_index(drop=True)
    
    candidate_nodes = list(solutions_reg.columns)[3:]
    counts={}
    for node in candidate_nodes:
        counts[node]=0

    if remove_duplicates:
        for i in range(solutions_reg_reduced1.shape[0]):
            for j in range(len(candidate_nodes)):
                if abs(solutions_reg_reduced1.iloc[i,j+3])>thres_val:
                    counts[candidate_nodes[j]]+=1

        freqs = {k:v/solutions_reg_reduced1.shape[0] for k,v in counts.items()}
    else:
        for i in range(solutions_reg_reduced2.shape[0]):
            for j in range(len(candidate_nodes)):
                if abs(solutions_reg_reduced2.iloc[i,j+3])>thres_val:
                    counts[candidate_nodes[j]]+=1

        freqs = {k:v/solutions_reg_reduced2.shape[0] for k,v in counts.items()}
        
    zero_cols = [k for k,v in counts.items() if v==0]
    non_zero_cols = [k for k,v in counts.items() if v>0]
    for col in zero_cols:
        freqs.pop(col)
    if remove_duplicates:
        return freqs, solutions_reg_reduced1[non_zero_cols].copy()
    else:
        return freqs, solutions_reg_reduced2[non_zero_cols].copy()

# Example Usage
#freqs, intv_vals = visualize_reg(solutions_reg, solutions_reg_processed, thres_val=0.1)
#freqs = sort_nodes(freqs)
#dowhy_plot.bar_plot(freqs, ylabel='Frequencies')
#sns.heatmap(intv_vals, cmap='coolwarm', center=0)