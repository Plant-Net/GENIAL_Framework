import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import decoupler as dc
import networkx as nx
from matplotlib import colors
from matplotlib.cm import ScalarMappable

def list_to_merc_f(gene_list, merc_res): 
    """
    This function will takes in input a gene list of interest and the mercator annotations (generally annotations on all the CDS of the 
    plant of interest) and will extract, from the annotations, only those related to the genes in the list 
    """
    hit_merc_res = merc_res[~(merc_res["IDENTIFIER"].isna())]
    pattern = "|".join(list(map(str.lower,gene_list))) # create a regex for str.contains for the different version of a gene .1, .2 etc...
    gene_hit = hit_merc_res[hit_merc_res["IDENTIFIER"].str.contains(pattern)]

    return gene_hit

def process_annotation(annot, grn):
    annot=list_to_merc_f(grn['target'].to_list(), annot)
    annot['NAME'] = annot['NAME'].str.split(".").str.get(0)
    annot['IDENTIFIER'] = annot['IDENTIFIER'].str.capitalize()
    annot.rename(columns={"NAME": "ANNOT"}, inplace=True)
    annot=annot[['IDENTIFIER', 'ANNOT']]
    annot_remove_unassigned=annot[annot["ANNOT"] != 'not assigned']
    return annot_remove_unassigned

def plot_annotation_distribution(annot):
    plot=annot['ANNOT'].value_counts(normalize=True)*100
    plt.figure(figsize=(10,6))
    sns.barplot(x=plot.index, y=plot.values, color='blue')
    plt.xticks(rotation=90)
    plt.ylabel('Percentage (%)')
    plt.title('Distribution of Mercator Annotations')
    plt.tight_layout()
    plt.show()
    
def import_and_process_dea(file_path, grn):
    stats_dea = pd.read_csv(file_path, sep='\t', index_col=0)
    stats_dea.reset_index(inplace=True)
    stats_dea['index'] = stats_dea['index'].str.split(':').str.get(1).str.split('.').str.get(0)
    stats_dea.set_index('index', inplace=True)
    stats_dea.columns = stats_dea.columns.str.replace('stat_', '', regex=False)
    stat_genes = stats_dea[stats_dea.index.isin(grn['target'].to_list()) | stats_dea.index.isin(grn['source'].to_list())]
    
    return stat_genes

def import_and_process_acts(file_path, grn):
    info = pd.read_csv(file_path, sep='\t', index_col=0)
    acts = info.filter(like='acts')
    acts.columns = acts.columns.str.replace('_acts', '', regex=False)
    acts = acts[acts.index.isin(grn['source'].to_list())]
    pval_acts = info.filter(like='pval')
    pval_acts.columns = pval_acts.columns.str.replace('_pval', '', regex=False)
    pval_acts = pval_acts[pval_acts.index.isin(grn['source'].to_list())]
    acts.iloc[:, 0:] = np.where(pval_acts.iloc[:, 0:] <= 0.05, acts.iloc[:, 0:], 0)
    acts = acts.loc[(acts != 0).any(axis=1)]
    return acts

def plot_heatmap_annotation(annot, grn):
    annot_remove_unassigned=annot[annot["ANNOT"] != 'not assigned']
    merged = grn.merge(annot_remove_unassigned, left_on='target', right_on='IDENTIFIER')
    tf_pathway_counts = merged.groupby(['source', 'ANNOT']).size().reset_index(name='count')
    heatmap_data = tf_pathway_counts.pivot(index='ANNOT', columns='source', values='count').fillna(0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="magma_r", annot=True, linewidths=.5, cbar_kws={'label': 'Number of Targets'}, fmt=".0f")
    plt.title('Number of TF Targets Annotated by Mercator')
    plt.xlabel('TF')
    plt.ylabel('Annotation')
    plt.tight_layout()
    plt.show()
    
def process_pathway_network(stats,annot):
    network_pathway_mercator = annot.copy()
    network_pathway_mercator.rename(columns={"IDENTIFIER": "target", "ANNOT":'source'}, inplace=True)
    network_pathway_mercator=network_pathway_mercator.drop_duplicates()
    matrix_stats=stats[stats.index.isin(annot['IDENTIFIER'].to_list())]
    matrix_stats = matrix_stats.T
    return network_pathway_mercator, matrix_stats


def compute_tf_pathway_association(tf_target_net, pathway_target_net, tf_activity, pathway_activity, 
                                   method='mean', min_shared_targets=5):
    """
    Compute TF-Pathway associations based on shared targets and activity matrices.

    Methods
    -------
    - "mean" : per-condition mean of TF and pathway activities
    - "correlation" : correlation across conditions (same value for all conditions)
    - "sign" : +1 if same sign, -1 if opposite sign, 0 if one or both are 0
    """
    triplet_list = []

    for tf in tf_activity.index:
        tf_targets = tf_target_net.loc[tf_target_net['source'] == tf, 'target'].tolist()

        # Candidate pathways via shared targets
        candidate_pathways = pathway_target_net[pathway_target_net['target'].isin(tf_targets)]['source'].unique()
        pathways = [pw for pw in candidate_pathways if pw in pathway_activity.index]
        if not pathways:
            continue

        for pw in pathways:
            pw_targets = pathway_target_net.loc[pathway_target_net['source'] == pw, 'target'].tolist()
            shared_targets = set(tf_targets) & set(pw_targets)

            if len(shared_targets) < min_shared_targets:
                continue

            # Decide association method
            if method == "correlation":
                tf_series = tf_activity.loc[tf]
                pw_series = pathway_activity.loc[pw]
                if tf_series.nunique() > 1 and pw_series.nunique() > 1:
                    assoc_val = np.corrcoef(tf_series, pw_series)[0, 1]
                else:
                    assoc_val = np.nan
                assoc_per_condition = {cond: assoc_val for cond in tf_activity.columns}

            elif method == "mean":
                assoc_per_condition = {
                    cond: (tf_activity.loc[tf, cond] + pathway_activity.loc[pw, cond]) / 2
                    for cond in tf_activity.columns
                }

            elif method == "sign":
                assoc_per_condition = {}
                for cond in tf_activity.columns:
                    tf_val = tf_activity.loc[tf, cond]
                    pw_val = pathway_activity.loc[pw, cond]
                    if tf_val == 0 or pw_val == 0:
                        assoc_per_condition[cond] = 0
                    elif (tf_val > 0 and pw_val > 0) or (tf_val < 0 and pw_val < 0):
                        assoc_per_condition[cond] = 1
                    else:
                        assoc_per_condition[cond] = -1

            else:
                raise ValueError(f"Unknown method: {method}")

            # Append results (shared for all methods)
            for cond in tf_activity.columns:
                triplet_list.append({
                    "TF": tf,
                    "Pathway": pw,
                    "Condition": cond,
                    "TF_Activity": tf_activity.loc[tf, cond],
                    "Pathway_Activity": pathway_activity.loc[pw, cond],
                    "Association": assoc_per_condition[cond],
                    "Shared_Targets": len(shared_targets)
                })

    return pd.DataFrame(triplet_list)


def plot_tf_pathway_network(triplet_df, condition, vmin=-4, vmax=4, cmap="coolwarm", association_mode=None):
    """
    Plot a TF-Pathway bipartite network for a given condition.

    Parameters
    ----------
    triplet_df : pd.DataFrame
        Must contain: ['TF', 'Pathway', 'Condition', 'TF_Activity', 'Pathway_Activity', 'Association']
    condition : str
        Condition to visualize.
    vmin, vmax : float
        Min/max for colormap normalization (for node colors).
    cmap : str or matplotlib colormap
        Colormap for node coloring.
    association_mode : str or None
        If "sign", edges are colored by association (-1=red, 0=gray, +1=green).
        Otherwise, edges are drawn in black.
    """
    df_cond = triplet_df[triplet_df['Condition'] == condition]
    if df_cond.empty:
        print(f"No data for condition: {condition}")
        return None

    # Build graph
    G = nx.Graph()
    tfs = df_cond['TF'].unique()
    pws = df_cond['Pathway'].unique()
    edges = df_cond[['TF', 'Pathway', 'Association']].drop_duplicates().values.tolist()
    edge_shared = df_cond.groupby(["TF", "Pathway"])["Shared_Targets"].max().to_dict()
    
    G.add_nodes_from(tfs, type="TF")
    G.add_nodes_from(pws, type="Pathway")
    G.add_edges_from([(e[0], e[1]) for e in edges])
    
    # Edge widths
    edge_widths = []
    for u, v in G.edges():
        val = edge_shared.get((u, v), edge_shared.get((v, u), 1))
        edge_widths.append(0.5 + val * 0.3)  # adjust scaling factor
        
    # Layout
    pos = nx.kamada_kawai_layout(G)

    # Colormap for nodes
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Activity values
    tf_values = df_cond.groupby("TF")["TF_Activity"].mean()
    pw_values = df_cond.groupby("Pathway")["Pathway_Activity"].mean()

    tf_colors = {tf: cmap(norm(tf_values.get(tf, 0))) for tf in tfs}
    pw_colors = {pw: cmap(norm(pw_values.get(pw, 0))) for pw in pws}

    # Edge colors
    if association_mode == "sign":
        assoc_map = dict(((row.TF, row.Pathway), row.Association) for _, row in df_cond.iterrows())
        edge_colors = []
        for u, v in G.edges():
            assoc = assoc_map.get((u, v), assoc_map.get((v, u), 0))
            if assoc == 1:
                edge_colors.append("blue")
            elif assoc == -1:
                edge_colors.append("red")
            else:
                edge_colors.append("lightgray")
    else:
        edge_colors = "black"
    
    if all(c == 'lightgray' for c in edge_colors):
        print(f"No informative associations for condition: {condition}")
        return None
    # Plot
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_title(f"TF-Pathway Network - Condition: {condition}", fontsize=14)

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=tfs,
                            node_color=[tf_colors[n] for n in tfs],
                            node_shape="^", node_size=600, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=pws,
                            node_color=[pw_colors[n] for n in pws],
                            node_shape="o", node_size=600, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        # Node colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Activity", rotation=270, labelpad=15)

        # Edge color legend (only if sign mode)
        if association_mode == "sign":
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color="blue", lw=2, label="Same sign (+1)"),
                Line2D([0], [0], color="red", lw=2, label="Opposite sign (-1)"),
                Line2D([0], [0], color="lightgray", lw=2, label="Zero (0)"),
            ]
            ax.legend(handles=legend_elements, title="Association", loc="upper left")

        plt.tight_layout()

    return fig
