### Retrieve GRN from gene list using the knowledge graph

from neo4j import GraphDatabase
import neo4j
import pandas as pd
import re
import argparse
print(neo4j.__version__)

# Connect to Neo4j database
uri = "bolt://127.0.0.1:7687"
username = "neo4j"
password = "neo4jpassword"
driver = GraphDatabase.driver(uri, auth=(username, password))
driver.verify_connectivity()

parser = argparse.ArgumentParser(description="Explore Hive list")
parser.add_argument('-f', '--file', type=str, help="Path to the gene list to analyze")
# parser.add_argument('-on','--output_net', type=str, help="Path to the output network file")
# parser.add_argument('-oi','--output_info', type=str, help="Path to the output information file")


# Define a function to extract the relevant part of the gene name
def extract_gene_name(gene_name):
    match = re.search(r'(Solyc\d+g\d+)', gene_name)
    return match.group(1) if match else None

# Define a function to read genes of interest from file, filter by cluster, and modify gene names
def read_genes_of_interest(file_path):
    genes_of_interest = []
    with open(file_path, "r") as file:
        # next(file)  # Skip the header line
        for line in file:
            parts = line.strip().split("\t")
            gene_name = extract_gene_name(parts[0])
            if gene_name:  # Ensure that the extraction was successful
                genes_of_interest.append(gene_name)
    return genes_of_interest


# Read genes of interest from file and perform modification
args = parser.parse_args()
HIVE = read_genes_of_interest(args.file)
# output_net = args.output_net
# output_info = args.output_info
#Query the Graph
def query_graph(gene_list):
    #Query to recover a GRN composed only of HIVE genes with associated description
    query_GRN_HIVE = f"""
    MATCH (tf:TranscriptionFactor)-[reg:Regulates]->(target:Gene)
    WHERE tf.name IN $gene AND target.name IN $gene
    RETURN tf.name, target.name, reg.evidence;
    """
    panda_res_GRN = driver.execute_query(
        query_GRN_HIVE,
        gene=gene_list,
        database_="neo4j",
        result_transformer_=neo4j.Result.to_df
    )
    
    query_genes_description = f"""
    MATCH (gene:Gene)
    WHERE gene.name IN $gene
    RETURN gene.name, gene.description, gene.family;
    """
    
    panda_description = driver.execute_query(
        query_genes_description,
        gene=gene_list,
        database_="neo4j",
        result_transformer_=neo4j.Result.to_df
    )
    
    query_pathway_background = f"""
    MATCH (gene:Gene)-[path:Involved_in]->(pathway:Pathway)
    WHERE gene.name IN $gene
    RETURN gene.name, pathway.name;
    """
    
    panda_pathway = driver.execute_query(
        query_pathway_background,
        gene=gene_list,
        database_="neo4j",
        result_transformer_=neo4j.Result.to_df
    )
    
    return panda_description, panda_res_GRN, panda_pathway

def export_network_and_description():
    description,network, pathways = query_graph(HIVE)
    description.to_csv('Data/Network_GRN_HIVE_INFO.txt', sep='\t', index=False)
    network.to_csv('Data/Network_GRN_HIVE.txt', sep='\t', index=False)
    # pathways.to_csv('Data/Mercator_background.txt', sep='\t', index=False)

if __name__ == "__main__":
    export_network_and_description()