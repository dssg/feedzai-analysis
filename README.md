# Application Grounded Evaluation of Explainable ML methods 

[![DOI](https://zenodo.org/badge/434338441.svg)](https://zenodo.org/badge/latestdoi/434338441)

This repository contains the code for the analyses we conducted for the paper titled "On the Importance of Application-Grounded Experimental Design for Evaluating Explainable ML Methods". It includes the performance metric calculations, error propagation, statistical tests, and cross validation across parameter ranges.

[The preprint of the manuscript is uploaded to arXiv](https://arxiv.org/abs/2206.13503)


## Replicating the analyses

The data used for calculating metrics and analyses is placed in `data/all_analyst_decisions.csv`.

1. The code in this repository is written to fetch data from a PostgreSQL database. However, we have exported the dataset in to a CSV for improving accessibility, and thus, the code needs to be adapted to match the new format. 

2. The code for calculating metrics is written in `src/analysis_functions.py`. The code includes several metrics but the metrics we used in our analysis for the manuscript are defined in the functions `pdr(...)` and `dt(...)`.


### Adapting the code to the CSV

The metric calculation functions and other analysis functions require the data to be loaded into a pandas DataFrame. The CSV can be loaded to a dataframe using `pandas.read_csv(...)` function. 

The column names we have used in our code and the ones in the CSV are different. Below is the mapping between the different versions. The dataframe should reflect the following column names.


```
Transaction ID: xplz_id
Transaction Amount (USD): trx_amnt
Fraud Label: label
Analyst ID: user_name
Experimental Arm: group
Analyst Decision: decision
Decition Time: decision_time
```
For analysing the responses to the questions, the notebook at `src/notebooks/question_answes.ipynb` can be used as a guide. 



