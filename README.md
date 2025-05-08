# WGCNA-in-python
Weighted correlation network analysis, also known as weighted gene co-expression network analysis (WGCNA), is a widely used data mining method especially for studying biological networks based on pairwise correlations between variables

# WGCNA-inspired Pipeline

This repository provides a customizable and efficient pipeline for constructing gene co-expression networks from expression data (CSV/TSV), performing correlation analysis, clustering, network construction, centrality analysis, and optional GO enrichment and Cytoscape export.

---

## Features

- Load gene expression matrix (`.csv` or `.tsv`)
- Filter genes by expression threshold and variance
- Parallelized correlation matrix calculation (Pearson, Spearman, Kendall)
- Plot interactive heatmaps and dendrograms
- Construct gene co-expression network with user-defined correlation threshold
- Analyze network centrality metrics
- Save `.graphml` file for Cytoscape visualization
- Perform Gene Ontology (GO) enrichment for detected communities (internet connection needed)

---

## Requirements

Install dependencies via `pip`:

```bash
pip install polars pandas numpy matplotlib seaborn networkx scipy gseapy plotly

