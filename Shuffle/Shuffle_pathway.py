import pandas as pd
import numpy as np
import decoupler as dc
import os
import multiprocessing as mp
import igraph as ig
import random
from scipy.stats import median_abs_deviation


def randomize_prior_igraph(prior_df, n_rewire=10_000, seed=42):
    """
    Degree-preserving randomization using igraph.
    Returns a randomized prior with the same in/out degrees.
    """
    random.seed(seed)
    
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


def compute_summary_statistics(real_scores, null_df):
    """
    Compute z-scores, empirical p-values, and other summary stats for each TF.
    """
    tf_stats = []

    for tf in real_scores['source'].unique():
        real_val = real_scores.loc[real_scores['source'] == tf, 'score'].mean()
        null_vals = null_df.loc[null_df['source'] == tf, 'score']
        
        if len(null_vals) == 0:
            continue  

        mu, sigma = null_vals.mean(), null_vals.std()
        z = (real_val - mu) / sigma if sigma > 0 else np.nan
        p_emp = (np.sum(np.abs(null_vals - mu) >= np.abs(real_val - mu)) + 1) / (len(null_vals) + 1) #two-sided empirical p-value
        significant = p_emp <= 0.05

        tf_stats.append({
            'TF': tf,
            'observed_score': real_val,
            'null_mean': mu,
            'null_std': sigma,
            'z_score': z,
            'empirical_p_value': p_emp,
            'significant': significant,
        })

    return pd.DataFrame(tf_stats)

def flatten_activity_df(df, cond_name=None, iteration=None):
    """
    Ensure correct orientation and convert to long format.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Wrong format.")

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

def list_to_merc_f(gene_list, merc_res): 
    """
    This function will takes in input a gene list of interest and the mercator annotations (generally annotations on all the CDS of the 
    plant of interest) and will extract, from the annotations, only those related to the genes in the list 
    """
    hit_merc_res = merc_res[~(merc_res["IDENTIFIER"].isna())]
    pattern = "|".join(list(map(str.lower,gene_list))) # create a regex for str.contains for the different version of a gene .1, .2 etc...
    gene_hit = hit_merc_res[hit_merc_res["IDENTIFIER"].str.contains(pattern)]

    return gene_hit

if __name__ == "__main__":
    import os
    os.makedirs("./results", exist_ok=True)
    
    n_random = 1000
    seed = 42

    # Load and process data
    prior_df = pd.read_csv("../Data/Network_GRN_HIVE_curated.txt", sep='\t')
    mercator_res = pd.read_csv("../Data/Mercator_annotation_Sly_4_1.txt", sep="\t", quotechar="'")
    
    expr_df = pd.read_csv("../DEA/Merge_stat_all.tsv", index_col=0, sep='\t')  # genes × conditions
    prior_df = process_prior(prior_df)
    expr_df = process_expr(expr_df, prior_df)
    
    mercator_res["IDENTIFIER"]= mercator_res["IDENTIFIER"].str.split(".").str.get(0)
    annot_target_grn=list_to_merc_f(prior_df['target'].to_list(), mercator_res)
    annot_target_grn['NAME'] = annot_target_grn['NAME'].str.split(".").str.get(0)
    annot_target_grn['IDENTIFIER'] = annot_target_grn['IDENTIFIER'].str.capitalize()
    annot_target_grn.rename(columns={"NAME": "ANNOT"}, inplace=True)
    annot_target_grn=annot_target_grn[['IDENTIFIER','ANNOT']]
    annot_remove_unassigned=annot_target_grn[annot_target_grn["ANNOT"] != 'not assigned']
    
    network_pathway_mercator = annot_remove_unassigned.copy()
    network_pathway_mercator.rename(columns={"IDENTIFIER": "target", "ANNOT":'source'}, inplace=True)
    network_pathway_mercator=network_pathway_mercator.drop_duplicates()
    network_pathway_mercator = network_pathway_mercator[network_pathway_mercator['target'].isin(expr_df.index)]
    network_pathway_mercator = network_pathway_mercator[['source', 'target']]

    # Transpose to shape: conditions × genes (as expected by decoupler)
    matrix = expr_df.T.fillna(0)

    # REAL Pathway ACTIVITY
    print("Computing real TF activity...")
    real_raw, real_pval = dc.run_mlm(
        mat=matrix,
        net=network_pathway_mercator,
        source='source',
        target='target',
        weight=None,
        verbose=True
    )
    real_scores = flatten_activity_df(real_raw)
    real_pval = flatten_activity_df(real_pval)
    real_scores.to_csv("./results/real_activity_all_conditions_pathways.csv", index=False)
    real_pval.to_csv("./results/real_pvalues_all_conditions_pathways.csv", index=False)

    # RANDOMIZED Pathway ACTIVITY (NULL)
    def run_random_iteration(i):
        rand_prior = randomize_prior_igraph(network_pathway_mercator, seed=seed + i)
        rand_raw, _ = dc.run_mlm(
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

    print(f"Running {n_random} random shuffles...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        null_results = pool.map(run_random_iteration, range(n_random))

    null_df = pd.concat(null_results)
    null_df.to_csv("./results/null_distribution_all_conditions_pathways.csv", index=False)

    # SUMMARY STATISTICS
    print("Writing summary statistics...")
    all_summary = []
    for cond in matrix.index:
        real_c = real_scores[real_scores['condition'] == cond]
        null_c = null_df[null_df['condition'] == cond]
        z_df = compute_summary_statistics(real_c, null_c)
        z_df['condition'] = cond
        all_summary.append(z_df)

    summary_df = pd.concat(all_summary)
    summary_df.to_csv("./results/tf_summary_all_conditions_pathways.csv", index=False)