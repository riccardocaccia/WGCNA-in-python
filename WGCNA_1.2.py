import argparse
import os
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.algorithms import community
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import gseapy as gp
import multiprocessing as mp
import plotly.express as px
import plotly.figure_factory as ff

class MethodError(Exception):
    '''personalized call error for wrong model choice'''
    def __init__(self, method):
        self.method = method
        message = f"Invalid method: '{method}'. Choose 'pearson' | 'spearman' | 'kendall'."
        super().__init__(message)

class AbsolutePath:
    '''class for useful paths'''
    def __init__(self, basename):
        self.basename = basename
        self.output_dir = f'{basename}_WGCNA_output'
        os.makedirs(self.output_dir, exist_ok=True)

def load_expression_data(expression_file):
    '''Load expression data from CSV or TSV based on extension'''
    ext = os.path.splitext(expression_file)[1].lower()
    match ext:
        case '.tsv':
            df = pl.read_csv(expression_file, separator='\t')
        case '.csv':
            df = pl.read_csv(expression_file)
        case _:
            raise ValueError(f'Unsupported file format: {ext}')
    return df

def subset_high_variance_genes(df: pl.DataFrame, top_n: int):
    '''filtering genes ...'''
    if df.shape[0] == 0:
        raise ValueError('no genes after filtering.')
    
    numeric_cols = df.columns[1:]

    #mean and sum of squares to calculate var
    df = df.with_columns([                                                    
        pl.mean_horizontal([pl.col(c) for c in numeric_cols]).alias('mean_expr'),
        pl.sum_horizontal([(pl.col(c) ** 2) for c in numeric_cols]).alias('sum_squares')
    ])

    n = len(numeric_cols)

    #variance
    df = df.with_columns(
        ((pl.col('sum_squares') / n) - (pl.col('mean_expr') ** 2)).alias('row_var') 
    )
    top_genes = df.sort('row_var', descending=True).head(top_n)

    if top_genes.is_empty():
        raise ValueError("No genes selected after variance filtering.")
    
    #returning the df without the temporary columns
    return top_genes.drop(["mean_expr", "sum_squares", "row_var"])

def remove_least_expressed_genes(expression_dataset: pl.DataFrame, expression_threshold: float):
    '''remove Least expressed gene based on a given threshold'''
    expr_no_na = expression_dataset.fill_null(0)
    numeric_cols = expr_no_na.columns[1:]
    expr_with_mean = expr_no_na.with_columns(
        pl.mean_horizontal([pl.col(c) for c in numeric_cols]).alias("mean_expr")
    )
    filtered = expr_with_mean.filter(pl.col("mean_expr") >= expression_threshold)
    return filtered.drop("mean_expr")

def _correlate_chunk(args):
    '''private function for chunk'''
    data, method, start, end = args  #numpyarray, method and chunck size = args
    result = []
    for i in range(start, end):  #each i is a gene
        row_i = data[i]
        row_corr = []
        for j in range(len(data)):
            if method == 'pearson':
                corr = np.corrcoef(row_i, data[j])[0, 1]
            elif method == 'spearman':
                corr = pd.Series(row_i).corr(pd.Series(data[j]), method='spearman')
            elif method == 'kendall':
                corr = pd.Series(row_i).corr(pd.Series(data[j]), method='kendall')
            else:
                raise MethodError(method)
            row_corr.append(corr)
        result.append(row_corr)
    return result

def parallel_correlation(df: pd.DataFrame, method: str, n_processes: int):
    '''parallel division of the matrix'''
    df = df.set_index(df.columns[0])
    data = df.to_numpy()
    n_genes = data.shape[0]
    chunk_size = max(1, n_genes // (n_processes or mp.cpu_count()))

    args = [
        (data, method, i, min(i + chunk_size, n_genes))
        for i in range(0, n_genes, chunk_size)
    ]
    
    #call of the private function
    with mp.Pool(processes=n_processes or mp.cpu_count()) as pool:
        chunks = pool.map(_correlate_chunk, args)
    
    corr_matrix = np.vstack(chunks)
    corr_df = pd.DataFrame(corr_matrix, index=df.index, columns=df.index)
    corr_df = corr_df.round(2).dropna(axis=0, how='all').dropna(axis=1, how='all')
    return corr_df

def correlation_heatmap(correlation_matrix: pd.DataFrame, basename_obj: AbsolutePath, output_dir):
    '''Draw an interactive correlation heatmap using Plotly'''
    print('drawing an heatmap...')

    fig = px.imshow(correlation_matrix.values, 
                    labels=dict(x="Genes", y="Genes", color="Correlation"),
                    x=correlation_matrix.columns, 
                    y=correlation_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    aspect="auto")
    fig.update_layout(
        title="Correlation Heatmap",
        xaxis=dict(tickangle=90, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
        autosize=True,
        height=800,) 

    fig.write_html(os.path.join(output_dir, f'{basename_obj.basename}_correlationHeatmap.html'))

    print(f'Interactive heatmap saved')

import plotly.figure_factory as ff
import numpy as np

#TODO: matrice triangolare con squareform
def dendrogram_plot(correlation_matrix, basename_obj: AbsolutePath, output_dir):
    '''Create an interactive dendrogram using Plotly and save as HTML'''

    distance_matrix = 1 - correlation_matrix
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # simmetria

    fig = ff.create_dendrogram(
        distance_matrix.values, 
        labels=correlation_matrix.columns.tolist(), 
        orientation='top',
        linkagefun=lambda x: linkage(squareform(x), method='average'))
    fig.update_layout(
        title="Gene Dendrogram",
        xaxis=dict(tickangle=90, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
        autosize=True,
        height=800,)

    fig.write_html(os.path.join(output_dir, f'{basename_obj.basename}_Dendrogram_Interactive.html'))

    print('Interactive dendrogram saved')

def build_network(correlation_matrix: pd.DataFrame, corr_threshold: float):
    '''build the conceptual network'''
    G = nx.Graph()
    G.add_nodes_from(correlation_matrix.columns)
    
    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            gene1 = correlation_matrix.index[i]
            gene2 = correlation_matrix.columns[j]
            correlation_value = correlation_matrix.iloc[i, j]
            if correlation_value >= corr_threshold:
                G.add_edge(gene1, gene2, weight=correlation_value)

    if G.number_of_edges() == 0:
        print('NESSUN ARCO ERRORE')
        return
    return G

def draw_network(network: nx.Graph, basename_obj: AbsolutePath, output_dir):
    '''draw the visual network'''
    pos = nx.spring_layout(network, seed=42)
    weights = [network[u][v]['weight'] for u, v in network.edges()]
    weight_range = np.linspace(1, 10, len(weights))
    edge_labels = nx.get_edge_attributes(network, 'weight')

    plt.figure(figsize=(10, 10))
    nx.draw(network, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=30, width=weight_range)
    nx.draw_networkx_edge_labels(network, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
    plt.title("Gene Co-Expression Network")
    plt.savefig(os.path.join(output_dir, f'{basename_obj.basename}_network_plot.png'), dpi=300)
    plt.close()

def centrality_analysis(network: nx.Graph, basename_obj: AbsolutePath, output_dir):
    '''centrality analysis'''
    largest_cc = max(nx.connected_components(network), key=len)
    subgraph = network.subgraph(largest_cc)
    
    degree_centrality = nx.degree_centrality(network)
    closeness_centrality = nx.closeness_centrality(network)
    eigen_centrality = nx.eigenvector_centrality(subgraph, max_iter=1000)
    clustering = nx.clustering(network)
    
    with open(os.path.join(output_dir, f'{basename_obj.basename}_centrality_analysis.txt'), 'w') as file:
        file.write("Degree Centrality:\n")
        for gene, centrality in degree_centrality.items():
            file.write(f"{gene}: {centrality:.2f}\n")
        file.write("\nCloseness Centrality:\n")
        for gene, centrality in closeness_centrality.items():
            file.write(f"{gene}: {centrality:.2f}\n")
        file.write("\nEigenvector Centrality:\n")
        for gene, centrality in eigen_centrality.items():
            file.write(f"{gene}: {centrality:.2f}\n")
        file.write("\nClustering Coefficient:\n")
        for gene, coeff in clustering.items():
            file.write(f"{gene}: {coeff:.2f}\n")
    
    return degree_centrality, closeness_centrality, eigen_centrality, clustering

def save_cytoscape_format(
    network: nx.Graph,
    basename_obj: AbsolutePath,
    output_dir: str,
    degree_centrality: dict,
    closeness_centrality: dict,
    eigen_centrality: dict,
    clustering: dict):
    '''save a graphml file with centrality metrics for Cytoscape'''
    print('Saving file for Cytoscape analysis...')
    # centrality metrics as node attributes
    for node in network.nodes():
        network.nodes[node]['degree_centrality'] = round(degree_centrality.get(node, 0), 2)
        network.nodes[node]['closeness_centrality'] = round(closeness_centrality.get(node, 0), 2)
        network.nodes[node]['eigen_centrality'] = round(eigen_centrality.get(node, 0), 2)
        network.nodes[node]['clustering_coefficient'] = round(clustering.get(node, 0), 2)

    nx.write_graphml(network, os.path.join(output_dir, f'{basename_obj.basename}.graphml'))
    print(f'Graphml file saved at wdir')


def get_gene_clusters(G):
    '''get gene clusters'''
    return [list(c) for c in community.louvain_communities(G)]  #list of list for cluster

def run_go_enrichment(gene_list, output_dir, cluster_id):
    '''conduct a GO enrichment'''
    try:
        gp.enrichr(
            gene_list=gene_list,
            gene_sets='GO_Biological_Process_2021',
            outdir=os.path.join(output_dir, f'cluster_{cluster_id}_GO'),
            cutoff=0.05
        )
    except Exception as i:
        print(f'[GO] Enrichment failed for cluster {cluster_id}: {i}')

def go_enrichment_all_clusters(G, output_dir):
    '''........'''
    clusters = get_gene_clusters(G)
    for i, gene_list in enumerate(clusters):
        run_go_enrichment(gene_list, output_dir, i)

#TODO: normalize optional!!
def get_args():
    '''parser arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument("expression_file", type=str, 
                        help="Path to gene expression matrix (.csv or .tsv)")
    parser.add_argument("--expression_threshold", type=float, default=1.0,
                        help='Expression level threshold for filtering genes')
    parser.add_argument("--correlation_threshold", type=float, default=0.85,
                        help='Correlation threshold to create edges in network')
    parser.add_argument("--method", type=str, choices=["pearson", "spearman", "kendall"], default="pearson",
                        help='Correlation method (pearson, spearman or kendall)')
    parser.add_argument("--top_genes", type=int, default=1000,
                        help='choose the chunck size and the top genes desired')
    parser.add_argument("--go_enrichment", type=str, choices=['yes','y','n','no'], default='no',
                        help='choose if you want to perform GO enrichment analysis')
    parser.add_argument("--n_processes", type=int, default=4, 
                        help="Number of processes to use to calculate the correlation")
    parser.add_argument("--cytoscape_file", type=str, choices=['yes', 'y', 'no', 'n'] , default='no',
                        help='y or n for saving a file for external cytoscape analysis')
    return parser.parse_args()

def main():
    #parser
    args = get_args()
    
    #path for the esperiment
    basename = AbsolutePath(os.path.splitext(os.path.basename(args.expression_file))[0])
    output_folder = basename.output_dir

    print('starting analysis...')

    #write dataframe into csv
    expression_dataset = load_expression_data(args.expression_file)
    expression_dataset = remove_least_expressed_genes(expression_dataset, args.expression_threshold)
    expression_dataset = subset_high_variance_genes(expression_dataset, args.top_genes)
    expression_dataset.write_csv(os.path.join(output_folder, f'{basename.basename}_filtered_expression.csv'))

    #correlation matrix
    correlation_matrix = parallel_correlation(expression_dataset.to_pandas(), 
                                              method=args.method, n_processes=args.n_processes).round(2)
    correlation_matrix.to_csv(os.path.join(output_folder, f'{basename.basename}_correlation_matrix.csv'), 
                              float_format="%.2f")
    ##Check
    if correlation_matrix.isnull().all().all(): #2 all for the 2D 
        raise ValueError("Only NAN")

    #graph
    correlation_heatmap(correlation_matrix, basename, output_folder)
    dendrogram_plot(correlation_matrix, basename, output_folder)

    #network se non è vuoto il tutto 
    network = build_network(correlation_matrix, args.correlation_threshold)
    if network:
        draw_network(network, basename, output_folder)
        degree, closness, eigen, cluster = centrality_analysis(network, basename, output_folder)

        #cytoscape file
        if args.cytoscape_file.lower() in ['yes', 'y']:
            save_cytoscape_format(network, basename, output_folder, degree, closness, eigen, cluster)


        #GO enrichment
        if args.go_enrichment.lower() in ['yes', 'y']:
            go_enrichment_all_clusters(network, output_folder)

    print('Analysis Finished')

if __name__ == "__main__":
    main()
