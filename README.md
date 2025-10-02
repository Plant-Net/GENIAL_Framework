# 🍅Tomato bulk RNA-seq GRN and TDA analysis

## Conditions :test_tube:
The conditions represents tomato infected with 7 different conditions (*Meloidogyne incognita* 7 and 14 dpi, *Botrytis cinerea*, *Phytophthora infestans*, *Cladosporium fulvum*, and *Potato spindle tuber viroid* mild and severe strains). Representing a total of 83 samples across 7 infections.
The data are publicaly available throught their respective bioproject:
<table>
    <tr>
        <th>Infection</th>
        <th>Tissue</th>
        <th>Controls hpi</th>
        <th>Controls replicates</th>
        <th>Infected hpi</th>
        <th>Infected Replicates</th>
        <th>BioProject</th>
        <th>Reference DOI</th>
    </tr>
    <tr>
        <td><i>M.incognita</i></td>
        <td>Root</td>
        <td>168</td>
        <td>8</td>
        <td>168</td>
        <td>8</td>
        <td>PRJNA734743</td>
        <td><a href="https://doi.org/10.3389/fpls.2022.817185">DOI</a></td>
    </tr>
    <tr>
        <td><i>M.incognita</i></td>
        <td>Root</td>
        <td>336</td>
        <td>8</td>
        <td>336</td>
        <td>8</td>
        <td>PRJNA734743</td>
        <td><a href="https://doi.org/10.3389/fpls.2022.817185">DOI</a></td>
    </tr>
    <tr>
        <td>PSTVd Mild strain</td>
        <td>Root</td>
        <td rowspan="2">408</td>
        <td rowspan="2">6</td>
        <td>408</td>
        <td>6</td>
        <td>PRJNA515609</td>
        <td><a href="https://doi.org/10.3390/v11110992">DOI</a></td>
    </tr>
    <tr>
        <td>PSTVd Severe strain</td>
        <td>Root</td>
        <td>408</td>
        <td>6</td>
        <td>PRJNA515609</td>
        <td><a href="https://doi.org/10.3390/v11110992">DOI</a></td>
    </tr>
    <tr>
        <td><i>B.cinerea</i></td>
        <td>Leaf</td>
        <td>0</td>
        <td>7</td>
        <td>30</td>
        <td>8</td>
        <td>PRJNA662936</td>
        <td><a href="https://doi.org/10.1093/plphys/kiab354">DOI</a></td>
    </tr>
    <tr>
        <td><i>P.infestans</i></td>
        <td>Leaf</td>
        <td>0</td>
        <td>6</td>
        <td>72</td>
        <td>6</td>
        <td>PRJNA505207</td>
        <td><a href="https://doi.org/10.1073/pnas.1814380116">DOI</a></td>
    </tr>
    <tr>
        <td><i>C.fulvum</i></td>
        <td>Leaf</td>
        <td>72</td>
        <td>3</td>
        <td>72</td>
        <td>3</td>
        <td>PRJNA781749</td>
        <td><a href="https://doi.org/10.3389/fgene.2023.1158631">DOI</a></td>
    </tr>
</table>

## Summary :compass:
From the multi-transcriptomics bulk RNA-seq data, we applied [HIVE](https://doi.org/10.1101/2024.03.04.583290). HIVE returned a list of genes available in the data folder but the following framework can be applied to any gene list.
From the list, we retrieved the GRN using **TomTom** neo4j database. We further curate the GRN to have a balance between confidence and sparsity.

We used decoupleR's ULM to infer TF activities and retrieve the significant ones. We consider the previous GRN and t-stat output of DESeq2 perform on each infection independently.
The analysis resulted in 43 significant regulatory TFs out of the 71 present in the GRN. We then used decoupleR's MLM to infer pathways activities from Mercator pathways (also available using TomTom).

Topological Data Analysis was performed on the same GRN with corresponding TF activities. We first applied the mapper algorithm to find a simpler representation of the GRN, and we further used the ToMATo algorithm to find groups on the mapper graph obtained before.

To find representatives nodes in each of the groups, we performed hub detection using degree and betweenness as metrics. We identified hubs as having the max value of either degree or betweenness in each group.

## Installation ⚙️
`conda create --name ENV_NAME python=3.12
pip install -r requirements.txt`

You also need R requirements such as DESEq2, edgeR, ggplot2, ... present in the R scripts.


## Run the framework :arrow_forward: 
You need the HIVE selection present in Data/ or any other matrix

From the raw transcriptomics table:
`DEA/DEA.R` to perform the DEA and get the needed Wald stats. Then `DEA/Merge_matrix.ipynb` to get the merged data used after.

GRN and Activities:
`GRN.ipynb` to retrieve the necessary networks (GRN) from TomTom and check them.
`TF_pathway_activity_mercator.ipynb` to perform TF and pathway activites.

TDA: 
`TDA/Prepare_data.ipynb` to format the data for TDA.
`TDA/mapper.py` to obtain the TDA network colored for all pathogens and the four configuration.
Finally, `TDA/Hub_TDA.ipynb` to detect hub in each of the configuration and `TDA/Pathway_TDA_and_link_TF_Pathway.ipynb` to check for TF - Pathway links in the sub-GRN of the hubs. 

For the plots, most of them are obtain with `Plot/Plot_clean.ipynb` or directly within dedicated notebook.

## Reference :writing_hand: 
You can find all the detailed and explained results [here](https://doi.org/10.1101/2025.04.09.647963)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15624949.svg)](https://doi.org/10.5281/zenodo.15624949)

