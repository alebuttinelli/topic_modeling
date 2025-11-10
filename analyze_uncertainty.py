# -*- coding: utf-8 -*-
"""
Script for uncertainty analysis of the BERTopic model.
Loads the probability matrix and metadata from a training 'run', 
identifies the documents with high uncertainty (low maximum probability or high entropy) and saves a CSV with these documents and their scores.

Example run:
    python analyze_uncertainty.py -r ./modelli_addestrati \
                                  -o ./analisi/documenti_incerti.csv \
                                  -t 0.4
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
from scipy.stats import entropy

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(run_dir: str, output_file: str, threshold: float):
    """
    Main function for analysing uncertainty
    """
    
    ## Define paths
    probs_path = os.path.join(run_dir, "probabilities.npy")
    metadata_path = os.path.join(run_dir, "metadata.parquet")

    ## Load
    logging.info("Loadings")
    try:
        probs = np.load(probs_path)
        meta_df = pd.read_parquet(metadata_path)
    except FileNotFoundError as e:
        logging.error(f"Error: File not found. {e}")
        return
    
    ## Identify uncertain documents
    
    # Creat df with probability distribution for every document
    df_probs = pd.DataFrame(probs)

    # Find the documents under the uncertain treshold
    uncertain_mask = df_probs.max(axis=1) < threshold
    uncertain_doc_indices = df_probs[uncertain_mask].index
    
    logging.info(f"Found {len(uncertain_doc_indices)} documents under the treshold of {threshold}.")

    if len(uncertain_doc_indices) == 0:
        logging.warning("No document under the uncertain treshold.")
        return

    ## Calculate entropy as a measure of uncertainty
    # Select only the uncertain documents
    uncertain_probs_array = probs[uncertain_doc_indices]

    # Calculate entropy
    uncertainty_scores = entropy(uncertain_probs_array.T)

    ## Output DataFrame preparation
    
    # Topic columns
    topic_columns = [f"Topic_{i}" for i in range(probs.shape[1])]

    # DataFrame definition
    df_uncertain_info = pd.DataFrame({
        "original_index": uncertain_doc_indices,
        "oggetti": meta_df.iloc[uncertain_doc_indices]["oggetto"].values,
        "uncertainty_score (entropy)": uncertainty_scores,
        "max_probability": df_probs.iloc[uncertain_doc_indices].max(axis=1).values
    })

    # DataFrame with probability distributionco
    df_uncertain_probs = pd.DataFrame(uncertain_probs_array, columns=topic_columns)

    df_final_output = pd.concat([df_uncertain_info, df_uncertain_probs], axis=1)

    # Order based on uncertainty (descending)
    df_final_output = df_final_output.sort_values(by="uncertainty_score (entropy)", ascending=False)

    ## Save the result
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    df_final_output.to_csv(output_file, index=False)
    logging.info(f"File saved in: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse the uncertainty of topic attribution of a BERTopic model.")
    
    parser.add_argument(
        "-r", 
        "--run_dir", 
        type=str, 
        required=True, 
        help="Input directory with 'probabilities.npy' and 'metadata.parquet'."
    )
    
    parser.add_argument(
        "-o", 
        "--output_file", 
        type=str, 
        required=True, 
        help=".csv file path to save the analysis."
    )
    
    parser.add_argument(
        "-t", 
        "--threshold", 
        type=float, 
        default=0.5,
        help="Uncertainty treshold"
    )
    
    args = parser.parse_args()
    main(args.run_dir, args.output_file, args.threshold)
