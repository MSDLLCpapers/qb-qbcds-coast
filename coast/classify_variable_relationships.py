
"""
This script classifies relationships between variables in a dataset into four categories:
1. Linear with Gaussian noise
2. Linear with Non-Gaussian noise
3. Non-Linear with Gaussian noise
4. Non-Linear with Non-Gaussian noise

To improve efficiency on large datasets, it analyzes each variable using only its top-k
most correlated predictors (default k=100), rather than all pairwise combinations.
This reduces computational complexity.

The script uses:
- Linear and polynomial regression to assess linearity
- R² scores to compare model fits
- D’Agostino and Pearson’s test to assess Gaussianity of residuals
- Parallel processing (via joblib) to speed up computation
"""



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import normaltest
from collections import Counter
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter

def classify_pair(x, y):
    lin_reg = LinearRegression().fit(x, y)
    y_pred_linear = lin_reg.predict(x)
    r2_linear = r2_score(y, y_pred_linear)
    residuals_linear = y - y_pred_linear

    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)
    poly_reg = LinearRegression().fit(x_poly, y)
    y_pred_poly = poly_reg.predict(x_poly)
    r2_poly = r2_score(y, y_pred_poly)
    residuals_poly = y - y_pred_poly

    _, p_linear = normaltest(residuals_linear)
    _, p_poly = normaltest(residuals_poly)

    is_linear = r2_linear >= 0.95 or r2_linear >= r2_poly
    is_gaussian = (p_linear > 0.05) if is_linear else (p_poly > 0.05)

    if is_linear and is_gaussian:
        return "Linear with Gaussian noise"
    elif is_linear and not is_gaussian:
        return "Linear with Non-Gaussian noise"
    elif not is_linear and is_gaussian:
        return "Non-Linear with Gaussian noise"
    else:
        return "Non-Linear with Non-Gaussian noise"

def classify_relationships_top_k(df, k=100):
    corr_matrix = df.corr().abs()
    columns = df.columns
    tasks = []

    for target in columns:
        top_k_predictors = corr_matrix[target].drop(target).nlargest(k).index
        for predictor in top_k_predictors:
            tasks.append((predictor, target))

    def process_pair(i, j):
        x = df[[i]].values
        y = df[j].values
        return classify_pair(x, y)

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(process_pair)(i, j) for i, j in tasks)

    category_counts = Counter(results)
    total = sum(category_counts.values())
    percentages = {key: f"{(value / total) * 100:.2f}%" for key, value in category_counts.items()}
    return percentages

################################################################################################################################
    
################
# Example usage:
################
# import time
# start = time.time()

# # rows:samples x columns:genes
# df = pd.read_csv("linear_gaussian_shared_noise_2000_variables.csv")
# results = classify_relationships_top_k(df, k=100)
# print(results)
# end = time.time()
# print(f"Wall time: {end - start:.6f} seconds")

