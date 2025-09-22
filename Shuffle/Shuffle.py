import pandas as pd
import numpy as np
import decoupler as dc
import os
import multiprocessing as mp
from functools import partial
import igraph as ig
from scipy.stats import median_abs_deviation


def randomize_prior_igraph(prior_df, n_rewire=10_000, seed=None):
    """
    Degree-preserving randomization using igraph.
    Returns a randomized prior with the same in/out degrees.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Map nodes to integers
    tf_set = prior_df['source'].unique()
    gene_set = prior_df['target'].unique()
    nodes = np.unique(np.concatenate([tf_set, gene_set]))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # Create edge list as tuples of integers
    edges = list(zip(
        prior_df['source'].map(node_to_idx),
        prior_df['target'].map(node_to_idx)
    ))
    
    g = ig.Graph(directed=True)
    g.add_vertices(len(nodes))
    g.add_edges(edges)

    # Rewire edges while preserving degree sequence
    g.rewire(mode="simple", n=n_rewire)

    # Extract randomized edges
    rand_edges = [(idx_to_node[e.source], idx_to_node[e.target]) for e in g.es]

    # Rebuild DataFrame
    rand_prior = pd.DataFrame(rand_edges, columns=["source", "target"])
    # print("Randomized prior columns:", rand_prior.columns)
    return rand_prior

# def is_iqr_outlier(real, null_vals):
#     q1 = np.percentile(null_vals, 25)
#     q3 = np.percentile(null_vals, 75)
#     iqr = q3 - q1
#     return real < (q1 - 1.5 * iqr) or real > (q3 + 1.5 * iqr)

# def is_mad_outlier(real, null_vals):
#     median = np.median(null_vals)
#     mad = median_abs_deviation(null_vals, scale='normal')
#     if mad == 0:
#         return False
#     mod_z = abs(real - median) / mad
#     return mod_z >= 3.5

def compute_summary_statistics(real_scores, null_df):
    """
    Compute z-scores, empirical p-values, and other summary stats for each TF.
    """
    tf_stats = []

    for tf in real_scores['source'].unique():
        real_val = real_scores.loc[real_scores['source'] == tf, 'score'].mean()
        null_vals = null_df.loc[null_df['source'] == tf, 'score']
        
        if len(null_vals) == 0:
            continue  # skip TFs not in null (rare)

        mu, sigma = null_vals.mean(), null_vals.std()
        z = (real_val - mu) / sigma if sigma > 0 else np.nan
        p_emp = (np.sum(np.abs(null_vals - mu) >= np.abs(real_val - mu)) + 1) / (len(null_vals) + 1) #two-sided empirical p-value
        significant = p_emp < 0.05

        tf_stats.append({
            'TF': tf,
            'observed_score': real_val,
            'null_mean': mu,
            'null_std': sigma,
            'z_score': z,
            'empirical_p_value': p_emp,
            'significant': significant
            # 'iqr_outlier': is_iqr_outlier(real_val, null_vals),
            # 'mad_outlier': is_mad_outlier(real_val, null_vals)
        })

    return pd.DataFrame(tf_stats)

def flatten_activity_df(df, cond_name=None, iteration=None):
    """
    Ensure correct orientation and convert to long format.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected DataFrame output from decoupler method.")

    # Transpose if TFs are columns instead of index
    if df.columns.str.startswith(('Solyc')).any():  # heuristic
        df = df.T

    df_flat = df.copy()
    df_flat.reset_index(inplace=True)
    df_flat.rename(columns={df_flat.columns[0]: 'source'}, inplace=True)

    df_flat = df_flat.melt(id_vars='source', var_name='condition', value_name='score')

    if cond_name is not None:
        df_flat['condition'] = cond_name
    if iteration is not None:
        df_flat['iteration'] = iteration

    return df_flat

def process_condition_from_merged(cond_name, expr_df, prior_df, n_random=100, seed=42):
    """
    Process one condition from merged expression matrix (gene t-stats).
    """
    print(f"Processing: {cond_name}")
    
    # Extract t-statistics for current condition
    cond_expr = expr_df[[cond_name]]
    cond_expr = cond_expr.fillna(0)  # Fill NaNs with 0
    cond_expr = cond_expr.T  # shape: 1 × genes
    # cond_expr.to_csv(f"./results/debug/cond_expr_{cond_name}.csv", index=False)
    
    # Real TF activity
    real_scores_raw, _ = dc.run_ulm(
    mat=cond_expr,
    net=prior_df,
    source='source',
    target='target',
    weight=None,
    verbose=True
    )
    
    real_scores = flatten_activity_df(real_scores_raw, cond_name=cond_name)

    # Null distribution via randomized priors
    null_scores = []
    for i in range(n_random):
        rand_prior = randomize_prior_igraph(prior_df, seed=seed + i)
        rand_raw, _ = dc.run_ulm(
        mat=cond_expr,
        net=rand_prior,
        source='source',
        target='target',
        weight=None,
        verbose=False
        )
        rand_activity = flatten_activity_df(rand_raw, cond_name=cond_name, iteration=i)
        null_scores.append(rand_activity)

    null_df = pd.concat(null_scores)
    # Compute summary statistics
    z_df = compute_summary_statistics(real_scores, null_df)

    # Save
    z_df.to_csv(f"./results/tf_summary_{cond_name}.csv", index=False)
    null_df.to_csv(f"./results/null_distribution_{cond_name}.csv", index=False)
    real_scores.to_csv(f"./results/real_activity_{cond_name}.csv", index=False)

    return cond_name

def process_prior(prior_df):
    """
    Process the prior DataFrame to ensure the column name.
    """
    prior_df = prior_df.rename(columns={'tf.name': 'source', 'target.name': 'target'})
    return prior_df

def process_expr(expr_df, prior_df):
    """
    Process the expression DataFrame to ensure it has the correct gene name.
    """
    expr_df = expr_df.reset_index()
    expr_df['OLN']=expr_df['index'].str.split(':').str.get(1).str.split('.').str.get(0)
    expr_df = expr_df[expr_df['OLN'].isin(prior_df['source']) | expr_df['OLN'].isin(prior_df['target'])]
    expr_df = expr_df.set_index('OLN')
    expr_df = expr_df.drop(columns=['index'])
    
    return expr_df

if __name__ == "__main__":
    import os
    os.makedirs("./results", exist_ok=True)
    
    n_random = 1000
    seed = 42

    # Load and process data
    prior_df = pd.read_csv("../Data/Network_GRN_HIVE_curated.txt", sep='\t')
    expr_df = pd.read_csv("../DEA/Merge_stat_all.tsv", index_col=0, sep='\t')  # genes × conditions
    prior_df = process_prior(prior_df)
    expr_df = process_expr(expr_df, prior_df)

    # Transpose to shape: conditions × genes (as expected by decoupler)
    matrix = expr_df.T.fillna(0)

    # REAL TF ACTIVITY
    print("🔍 Running real TF activity...")
    real_raw, _ = dc.run_ulm(
        mat=matrix,
        net=prior_df,
        source='source',
        target='target',
        weight=None,
        verbose=True
    )
    real_scores = flatten_activity_df(real_raw)
    real_scores.to_csv("./results/real_activity_all_conditions.csv", index=False)

    # RANDOMIZED TF ACTIVITY (NULL)
    def run_random_iteration(i):
        rand_prior = randomize_prior_igraph(prior_df, seed=seed + i)
        rand_raw, _ = dc.run_ulm(
            mat=matrix,
            net=rand_prior,
            source='source',
            target='target',
            weight=None,
            verbose=False
        )
        df_flat = flatten_activity_df(rand_raw)
        df_flat['iteration'] = i
        return df_flat

    print(f"🔁 Running {n_random} random shuffles with multiprocessing...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        null_results = pool.map(run_random_iteration, range(n_random))

    null_df = pd.concat(null_results)
    null_df.to_csv("./results/null_distribution_all_conditions.csv", index=False)

    # SUMMARY STATISTICS
    print("📊 Computing summary statistics...")
    all_summary = []
    for cond in matrix.index:
        real_c = real_scores[real_scores['condition'] == cond]
        null_c = null_df[null_df['condition'] == cond]
        z_df = compute_summary_statistics(real_c, null_c)
        z_df['condition'] = cond
        all_summary.append(z_df)

    summary_df = pd.concat(all_summary)
    summary_df.to_csv("./results/tf_summary_all_conditions.csv", index=False)

    print("✅ Benchmark complete.")