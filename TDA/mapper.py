import sys
import numpy as np
import gudhi as gd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.preprocessing import LabelEncoder
from gudhi.clustering.tomato import Tomato
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse.csgraph import connected_components
from gudhi.cover_complex import MapperComplex
import pickle as pck
import networkx as nx
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import math
from networkx.drawing.nx_agraph import graphviz_layout
from gudhi.cover_complex import MapperComplex
from patched_cover_complex import fit
MapperComplex.fit = fit

plt.rcParams['svg.fonttype'] = 'none'
# Clustering method based on GRN
class GraphCC(BaseEstimator, TransformerMixin):
    def __init__(self, graph):
        self.graph = graph
    def fit(self, X, y=None):
        return self
    def fit_predict(self, X, y=None):
        n_components, labels = connected_components(csgraph=self.graph[:,X[:,0]][X[:,0],:], directed=False)
        self.n_clusters_ = n_components
        self.labels_ = labels
        return self.labels_

# Read GRN
X = pd.read_csv('./Data/Curated_gene_regulatory_network.tsv', sep='\t', header=0, index_col=None)
X = np.array(X)
# print(len(X), len(np.unique(X[:,0])), len(np.unique(X[:,1])))

# Encoding names with integers
le = LabelEncoder()
n_tf, n_genes = len(np.unique(X[:,0])), len(np.unique(X[:,1]))
le.fit(np.array(list(np.unique(X[:,0])) + list(np.unique(X[:,1]))))

X[:,0] = le.transform(X[:,0])
X[:,1] = le.transform(X[:,1])

# Read expression + stats
F1 = pd.read_csv('./Data/LogFC_genes_in_cGRN.tsv', sep='\t', header=0, index_col=None)
F2 = pd.read_csv('./Data/Stats_genes_in_cGRN.tsv', sep='\t', header=0, index_col=None)
# print(np.array(F1), np.array(F2))
# print(F1.columns)

F1, F2 = np.array(F1), np.array(F2)
F1[:,0] = le.transform(F1[:,0])
F2[:,0] = le.transform(F2[:,0])
# print(F1,F2)

# Read activity
A = pd.read_csv('./Data/Activity_TF_in_cGRN.tsv', sep='\t', header=0, index_col=None)
pathogen_names = np.array(A.columns)
A = np.array(A)
A[:,0] = le.transform(A[:,0])

### TF-only GRN

# Restrict to TF
f1_tf,_,tf_indices = np.intersect1d(X[:,0], F1[:,0], return_indices=True)
F1_tf = F1[tf_indices,:]
f2_tf,_,tf_indices = np.intersect1d(X[:,0], F2[:,0], return_indices=True)
F2_tf = F2[tf_indices,:]
a_tf,_,tf_indices = np.intersect1d(X[:,0], A[:,0], return_indices=True)
A_tf = A[tf_indices,:]

le_tf = LabelEncoder()
le_tf.fit(X[:,0])

f1_perm = le_tf.transform(f1_tf)
inv_perm = np.array([np.argwhere(f1_perm == i)[0][0] for i in range(n_tf)]) 
F1_tf = F1_tf[inv_perm]
f2_perm = le_tf.transform(f2_tf)
inv_perm = np.array([np.argwhere(f2_perm == i)[0][0] for i in range(n_tf)]) 
F2_tf = F2_tf[inv_perm]
A_perm = le_tf.transform(a_tf)
inv_perm = np.array([np.argwhere(A_perm == i)[0][0] for i in range(n_tf)]) 
A_tf = A_tf[inv_perm]

common_tf = np.intersect1d(X[:,0], X[:,1])
targets_tf = [[] for _ in range(n_tf)]
graph_tf = np.zeros([n_tf, n_tf])
common_targets_tf = np.zeros([n_tf, n_tf])

# Construct networkx graph
for pair in X:
    tf0_id = le_tf.transform([pair[0]])[0]
    if pair[1] in common_tf:
        tf1_id = le_tf.transform([pair[1]])[0]
        graph_tf[tf0_id, tf1_id] = 1
        graph_tf[tf1_id, tf0_id] = 1
    targets_tf[tf0_id].append(pair[1])

# Compute common gene targets between TFs
for i in range(n_tf):
    for j in range(i+1,n_tf):
        common_targets_tf[i,j] = len(np.intersect1d(np.array(targets_tf[i]), np.array(targets_tf[j])))
        common_targets_tf[j,i] = len(np.intersect1d(np.array(targets_tf[i]), np.array(targets_tf[j])))

common_targets_tf_save = common_targets_tf

gene_names = le.inverse_transform((le_tf.inverse_transform(np.arange(common_targets_tf_save.shape[0]))).astype(int))

# Build sets of targets per TF (unique targets)
sets = [set(trg) for trg in targets_tf]
n_tf_calc = common_targets_tf_save.shape[0]

# Jaccard index (symmetric): intersection / union
jaccard = np.full((n_tf_calc, n_tf_calc), np.nan, dtype=float)
for i in range(n_tf_calc):
    for j in range(n_tf_calc):
        A, B = sets[i], sets[j]
        union_size = len(A | B)
        if union_size > 0:
            inter_size = len(A & B)
            jaccard[i, j] = inter_size / union_size

jaccard_df = pd.DataFrame(jaccard, index=gene_names, columns=gene_names)
annot = jaccard_df.round(3).astype(object).astype(str)

plt.figure(figsize=(50, 40))  # Adjust the figure size for better visibility
sns.heatmap(
    jaccard_df,
    cmap="magma_r",
    annot=annot,
    fmt='',
    cbar=True,
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Jaccard index'}
)


common_targets_tf = np.where(common_targets_tf > 0, np.ones(common_targets_tf.shape), np.zeros(common_targets_tf.shape))

#G = nx.Graph(graph_tf)
#plt.figure()
#nx.draw(G, node_size=1)
#plt.show()

# Initialize clustering based on GRNs
# Common targets-based clustering
grnCC_tf = GraphCC(common_targets_tf)
# GRN-based clustering
# grnCC_tf = GraphCC(graph_tf)

# Build filter out of graph Laplacian
n_targets_tf = np.array([len(trg) for trg in targets_tf])
L = nx.normalized_laplacian_matrix(nx.Graph(graph_tf))
eigenvalues, eigenvectors = np.linalg.eig(L.toarray())
eigenvectors = np.real(eigenvectors)
print(eigenvectors.shape)

#plt.figure()
#plt.hist(n_targets_tf, bins=300)
#plt.show()
#plt.figure()
#plt.hist(eigenvectors[:,0], bins=300)
#plt.show()
amin, amax = A_tf[:,1:].min(), A_tf[:,1:].max()

## Filter unsignificant TF for visualization of clusters
info = pd.read_csv('../Results_hubs_activity_stats_shuffle_corrected.tsv', sep='\t')
info = info[info['OLN'].isin(le.inverse_transform(A_tf[:,0].astype(int)))]
info = info.loc[:, ['OLN', *info.loc[:, 'Pinfestans_acts':'Mincognita_14dpi_acts'].columns, 'Groups_corrected']]
act_cols = [c for c in info.columns if c.endswith('_acts')]

def mask_inactive(row):
    group = row['Groups_corrected']
    new_row = row.copy()
    for col in act_cols:
        cond_name = col.replace('_acts', '_pval')
        if cond_name not in str(group):
            new_row[col] = np.nan
    return new_row

info_filtered = info.apply(mask_inactive, axis=1)
info_filtered['OLN'] = le.transform(info_filtered['OLN'])
A_tf_filtered = info_filtered[['OLN']+act_cols].to_numpy()
# (~np.isnan(A_tf_filtered[:,1:])).sum() ## correctly match the bool df from the shuffling

# A_tf_filtered_old = np.where(np.abs(A_tf) < 2, np.nan , A_tf)
# A_tf_filtered_old = A_tf_filtered_old.astype(float)
# A_tf_filtered_old[np.all(np.isnan(A_tf_filtered_old), axis=1)] = 0

# Compute Mapper
#print(n_tf, n_targets_tf[:,None].shape, np.arange(n_tf)[:,None].shape)
mapper = MapperComplex(input_type='point cloud', min_points_per_node=0, filter_bnds=None, resolutions=[5,5], gains=[0.3,0.3], clustering=grnCC_tf)
# mapper.fit(X=np.arange(n_tf)[:,None], filters=eigenvectors[:,0:2], colors=np.hstack([n_targets_tf[:,None], A_tf[:,1:], F1_tf[:,1:], F2_tf[:,1:]])) #n_targets_tf[:,None])

mapper.fit(X=np.arange(n_tf)[:,None], filters=eigenvectors[:,0:2], colors=np.hstack([n_targets_tf[:,None], A_tf_filtered[:,1:]]))

# Interactive visu
# mapper.save_to_html(file_name="TFs", data_name="GRN", cover_name="uniform", color_name="F1:1")

# 2D visu
mapper_graph = mapper.get_networkx()

# print([(v, le.inverse_transform(  [int(i) for i in le_tf.inverse_transform(mapper.node_info_[v]["indices"])]  )) for v in mapper_graph.nodes()])
with open("./mapper_nodes.txt", "w") as f:
    result = [(v, le.inverse_transform([int(i) for i in le_tf.inverse_transform(mapper.node_info_[v]["indices"])])) for v in mapper_graph.nodes()]
    f.write(str(result))

mapper_graph.remove_nodes_from(list(nx.isolates(mapper_graph)))

## Custom cmap for tf activity
cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)
# cmap = sns.color_palette('coolwarm', as_cmap=True)
for cond in range(1, A_tf_filtered.shape[1]):
    plt.figure()
    node_colors = [mapper.node_info_[v]["colors"][cond] for v in mapper_graph.nodes()]
    # Specify node sizes and edge weights
    # One possible layout, use the one you prefer
    pos = nx.kamada_kawai_layout(mapper_graph)
    # pos = graphviz_layout(mapper_graph, prog="dot")
    # Specify node colors
    pathcollection = nx.draw_networkx_nodes(mapper_graph, pos, node_color=node_colors, cmap=cmap, vmin=-2.5, vmax=2.5)
    # Specify node sizes and edge weights
    nx.draw(mapper_graph, pos=pos, with_labels=True,
                          node_color=node_colors,
                          cmap=cmap,
                          vmin=-2.5,
                          vmax=2.5,
                          node_size=[10*mapper.node_info_[v]["colors"][0] for v in mapper_graph.nodes()],
                          width=[  common_targets_tf[np.unique(np.concatenate([mapper.node_info_[e[0]]["indices"], mapper.node_info_[e[1]]["indices"]])),:]
                                                    [:,np.unique(np.concatenate([mapper.node_info_[e[0]]["indices"], mapper.node_info_[e[1]]["indices"]]))].sum()/100 for e in mapper_graph.edges()])
    # Plot colorscale
    plt.colorbar(pathcollection)
    plt.savefig('./Plot/mapper_filtered_' + str(cond) + '.svg', format='svg', dpi=300, bbox_inches='tight')   

### Plot Graph without coloring
plt.figure()
pos = nx.kamada_kawai_layout(mapper_graph)
nx.draw(mapper_graph, pos=pos, with_labels=True,
                        node_size=[10*mapper.node_info_[v]["colors"][0] for v in mapper_graph.nodes()],
                        width=[  common_targets_tf[np.unique(np.concatenate([mapper.node_info_[e[0]]["indices"], mapper.node_info_[e[1]]["indices"]])),:]
                                                    [:,np.unique(np.concatenate([mapper.node_info_[e[0]]["indices"], mapper.node_info_[e[1]]["indices"]]))].sum()/100 for e in mapper_graph.edges()])


## Get the different configurations
mapper_graph = mapper.get_networkx()
mapper_nodes = mapper_graph.nodes()
n_nodes = len(mapper_nodes)
mapper_neighbors = [[] for _ in range(n_nodes)]
perm = {idx_r: id_v for idx_r, id_v in enumerate(mapper_nodes)}
mapper_matrix = nx.adjacency_matrix(mapper_graph)
mapper_matrix = mapper_matrix.toarray()
for idx_r, id_v in enumerate(mapper_nodes):
    idx_neighb = np.argwhere(mapper_matrix[idx_r,:] == 1).ravel()
    neighb_v = [perm[n] for n in idx_neighb]
    mapper_neighbors[id_v] = neighb_v
mapper_width = np.zeros([n_nodes, n_nodes])
edge_width = []
for e in mapper_graph.edges():
    width = common_targets_tf[np.unique(np.concatenate([mapper.node_info_[e[0]]["indices"], mapper.node_info_[e[1]]["indices"]])),:][:,np.unique(np.concatenate([mapper.node_info_[e[0]]["indices"], mapper.node_info_[e[1]]["indices"]]))].sum()/100
    mapper_width[e[0],e[1]] = width
    mapper_width[e[1],e[0]] = width
    edge_width.append(width)
target_function = mapper_width.sum(axis=1)
tmin, tmax = target_function.min(), target_function.max()

#print([(v, le.inverse_transform(  [int(i) for i in le_tf.inverse_transform(mapper.node_info_[v]["indices"])]  )) for v in mapper_graph.nodes()])

def gaussian_1D(x,m,s):
    return np.exp(-(x-m)**2/s**2)

def activity_modes(x, s):
    return gaussian_1D(x, -4., s) + gaussian_1D(x, 0., s) + gaussian_1D(x, 4., s) if not np.isnan(x) else -10.

def target_modes(x, s):
    return gaussian_1D(x, 0., s) + gaussian_1D(x, 33., s)

a_bw, t_bw = .85, 10.

plt.figure()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(np.arange(-5.,5.,0.01), [activity_modes(x, a_bw) for x in np.arange(-5.,5.,0.01)])
plt.xlabel('Old Filter (TF ULM Activity)')
plt.ylabel('New Filter')
plt.savefig('./Plot/Filter_function_activity.svg', format='svg')
plt.show()

plt.figure()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(np.arange(0.,33.,0.1), [target_modes(x, t_bw) for x in np.arange(0.,33.,0.1)])
plt.xlabel('Old Filter (edge thickness)')
plt.ylabel('New Filter')
plt.savefig('./Plot/Filter_function_edge.svg', format='svg')
plt.show()

clusters, clusters_nodes, clusters_nodes_ens = {}, {v: [] for v in mapper_graph.nodes()}, {v: [] for v in mapper_graph.nodes()}

for cond in range(1, A_tf.shape[1] + 1):

    ##### Display graph

    plt.figure()
    # One possible layout, use the one you prefer
    pos = nx.kamada_kawai_layout(mapper_graph)
    # Specify node colors
    #node_color = [mapper.node_info_[v]["colors"][cond] for v in mapper_graph.nodes()] if cond < A_tf.shape[1] else [target_function[v] for v in mapper_graph.nodes()]
    #node_color = [activity_modes(mapper.node_info_[v]["colors"][cond], a_bw) for v in mapper_graph.nodes()] if cond < A_tf.shape[1] else [target_modes(target_function[v], t_bw) for v in mapper_graph.nodes()]
    #ncmin, ncmax = np.array(node_color).min(), np.array(node_color).max()
    pathcollection = nx.draw_networkx_nodes(mapper_graph, pos, 
                                            node_color=[mapper.node_info_[v]["colors"][cond] for v in mapper_graph.nodes()] if cond < A_tf.shape[1] else [target_function[v] for v in mapper_graph.nodes()], 
                                            vmin=amin if cond < A_tf.shape[1] else tmin, vmax=amax if cond < A_tf.shape[1] else tmax,
                                            #node_color=[activity_modes(mapper.node_info_[v]["colors"][cond], a_bw) for v in mapper_graph.nodes()] if cond < A_tf.shape[1] else [target_modes(target_function[v], t_bw) for v in mapper_graph.nodes()], 
                                            #vmin=ncmin, vmax=ncmax,
                                            #vmin=0, vmax=1,
                                            node_size=[30+10*mapper.node_info_[v]["colors"][0] for v in mapper_graph.nodes()])
    # Specify node sizes and edge weights
    nx.draw(mapper_graph, pos=pos, with_labels=True,
                          node_color=[np.nan for v in mapper_graph.nodes()],
                          width=edge_width)
    # Plot colorscale
    plt.colorbar(pathcollection)
    # plt.savefig('mapper_' + str(cond))

    ##### Node clustering

    tomato = Tomato(graph_type='manual', density_type='manual', n_clusters=None, merge_threshold=None)
    weights = [0 for _ in range(n_nodes)]
    for v in mapper_graph.nodes():
        weights[v] = activity_modes(mapper.node_info_[v]["colors"][cond], a_bw) if cond < A_tf.shape[1] else target_modes(target_function[v], t_bw)
    tomato.fit(X=mapper_neighbors, y=None, weights=weights)
    n_clusters = tomato.labels_.max() + 1

    ##### Display graph

    fig, ax = plt.subplots()
    # One possible layout, use the one you prefer
    pos = nx.kamada_kawai_layout(mapper_graph)
    # Specify node colors
    cmap = plt.get_cmap('rainbow', n_clusters)
    pathcollection = nx.draw_networkx_nodes(mapper_graph, pos, 
                                            node_color=[tomato.labels_[v] for v in mapper_graph.nodes()], 
                                            node_size=[100 if weights[v] != -10. else 0 for v in mapper_graph.nodes()],
                                            cmap=cmap)
    # Specify node sizes and edge weights
    nx.draw(mapper_graph, pos=pos, with_labels=True,
                          node_color=[np.nan for v in mapper_graph.nodes()],
                          width=edge_width)
    # Plot colorscale
    delta = (n_clusters-1)/n_clusters
    cbar = fig.colorbar(pathcollection, ticks=np.arange(delta/2, n_clusters-1, delta))
    cbar.ax.set_yticklabels(np.arange(0, n_clusters, 1))

    # plt.savefig('mapper_' + str(cond) + '_clusters')

    ### Display clusters

    for lab in np.unique(tomato.labels_):
        idx_lab = np.argwhere((tomato.labels_ == lab) & (np.array(weights) != -10.)).ravel() if cond < A_tf.shape[1] else np.argwhere(tomato.labels_ == lab).ravel()
        if cond < A_tf.shape[1]:
            clusters[pathogen_names[cond] + '_cluster' + str(lab)] = idx_lab
            for idx in idx_lab:
                clusters_nodes[idx].append(pathogen_names[cond] + '_cluster' + str(lab))
                pathogen = pathogen_names[cond].split('_')[0]
                if pathogen not in clusters_nodes_ens[idx]:
                    clusters_nodes_ens[idx].append(pathogen)
        else:
            clusters['numtargets_cluster' + str(lab)] = idx_lab
            for idx in idx_lab:
                clusters_nodes[idx].append('numtargets_cluster' + str(lab))
                if lab <= 5:
                    targetlab = 'exclusive'
                elif lab == 6: 
                    targetlab = 'mildly_shared'
                else:
                    targetlab = 'shared'
                clusters_nodes_ens[idx].append(targetlab)

#print(clusters)
for k, v in clusters_nodes_ens.items():
    if len(v) - 1 == 0:
        v = ['N/A'] + v
    elif len(v) - 1 == 1:
        v = ['specific'] + v[-1:]
    else:
        v = ['multiple'] + v[-1:]
    clusters_nodes_ens[k] = v

##### Display graph

mapper_graph.remove_nodes_from(list(nx.isolates(mapper_graph)))

clusters_ens = ['' for k,v in clusters_nodes_ens.items()]
for k,v in clusters_nodes_ens.items():
    clusters_ens[k] = '_'.join(v)
list_clusters = np.unique(clusters_ens)
n_clusters_ens = len(list_clusters)
clusters_env_perm = {idx_clus: clus for idx_clus, clus in enumerate(list_clusters)}
clusters_env_perm_inv = {clus: idx_clus for idx_clus, clus in enumerate(list_clusters)}

cluster_to_color={
    'specific_exclusive' : 'red',
    'multiple_shared' : 'green',
    'multiple_mildly_shared' : 'gold',
    'multiple_exclusive' : 'royalblue',
    'N/A_shared' : 'grey',
    'N/A_mildly_shared' : 'grey',
    'N/A_exclusive' : 'grey',
}
cmap = mcolors.ListedColormap([cluster_to_color[c] for c in list_clusters])

plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots()
# One possible layout, use the one you prefer
pos = nx.kamada_kawai_layout(mapper_graph)
# Specify node colors
# cmap = plt.get_cmap('rainbow', n_clusters_ens)
pathcollection = nx.draw_networkx_nodes(mapper_graph, pos, 
                                        node_color=[clusters_env_perm_inv[clusters_ens[v]] for v in mapper_graph.nodes()], 
                                        # node_size=[800 if clusters_ens[v].split('_')[0] != 'N/A' else 0 for v in mapper_graph.nodes()],
                                        node_size=[10*mapper.node_info_[v]["colors"][0] if clusters_ens[v].split('_')[0] != 'N/A' else 0 for v in mapper_graph.nodes()],
                                        cmap=cmap)
# Specify node sizes and edge weights
nx.draw(mapper_graph, pos=pos, with_labels=True,
                      node_color=[np.nan for v in mapper_graph.nodes()],
                      width=edge_width)
# Plot colorscale
delta = (n_clusters_ens-1)/n_clusters_ens
cbar = fig.colorbar(pathcollection, ticks=np.arange(delta/2, n_clusters_ens-1, delta))
cbar.ax.set_yticklabels([clusters_env_perm[c] for c in range(n_clusters_ens)])

plt.savefig('./Plot/Ensembl_cluster_TDA.svg', format='svg')
plt.show()