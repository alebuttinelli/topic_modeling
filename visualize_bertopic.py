# -*- coding: utf-8 -*-
"""
Script for visualizing a trained BERTopic model.

Loads a model, embeddings, topics, and metadata from a
"run" directory (created by train_bertopic.py) and generates
interactive HTML visualizations.

Example run:
    python visualize_bertopic.py -r ./trained_models -o ./visualizations
"""

import pandas as pd
import numpy as np
import torch
import argparse
import os
import logging
from bertopic import BERTopic
from umap import UMAP
from collections import defaultdict

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(run_dir: str, output_dir: str):
    """
    Main function to load data and generate visualizations.
    """
    
    ## Check and create the output directory
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    ## Define artifact paths (based on run_dir)
    model_path = os.path.join(run_dir, "bertopic_model")
    embeddings_path = os.path.join(run_dir, "embeddings.npy")
    topics_path = os.path.join(run_dir, "topics.npy")
    texts_path = os.path.join(run_dir, "texts.json") 
    metadata_path = os.path.join(run_dir, "metadata.parquet")
    topic_info_path = os.path.join(run_dir, "topic_info.csv")

    ## Load all artifacts
    logging.info("Loading model artifacts...")
    try:
        topic_model = BERTopic.load(model_path)
        embeddings = np.load(embeddings_path)
        topics = np.load(topics_path)
        texts = pd.read_json(texts_path).squeeze().tolist()
        meta_df = pd.read_parquet(metadata_path)
        topic_info_df = pd.read_csv(topic_info_path)
    except FileNotFoundError as e:
        logging.error(f"Error: File not found. {e}")
        logging.error(f"Ensure that '{run_dir}' contains all training artifacts.")
        return

    logging.info("All artifacts loaded successfully.")
    
    ## Generate Visualization 1: Visualize Topics
    logging.info("Generating 'visualize_topics' visualization...")
    fig_topics = topic_model.visualize_topics(custom_labels=True)
    out_path = os.path.join(output_dir, "visualize_topics.html")
    fig_topics.write_html(out_path)
    logging.info(f"Saving to {out_path}")

    ## Generate Visualization 2: Visualize Documents
    logging.info("Generating 'visualize_documents' visualization...")
    logging.info("... (Calculating UMAP reduction for visualization)...")
    
    reduced_embeddings = UMAP(n_neighbors=10, 
                              n_components=2, 
                              min_dist=0.0, 
                              metric='cosine').fit_transform(embeddings)
    
    nomi = meta_df["nome"].tolist()
    fig_docs = topic_model.visualize_documents(nomi, 
                                               reduced_embeddings=reduced_embeddings, 
                                               custom_labels=True)
    out_path = os.path.join(output_dir, "visualize_documents.html")
    fig_docs.write_html(out_path)
    logging.info(f"Saving to {out_path}")

    ## Generate Visualization 3: Topics per Class
    logging.info("Generating 'visualize_topics_per_class' visualization...")
    classes = meta_df['direzione'].tolist() # Use ALIGNED metadata
    topics_per_class = topic_model.topics_per_class(texts, classes=classes)
    
    fig_class = topic_model.visualize_topics_per_class(topics_per_class, 
                                                       top_n_topics=10, 
                                                       custom_labels=True)
    out_path = os.path.join(output_dir, "visualize_topics_per_class.html")
    fig_class.write_html(out_path)
    logging.info(f"Saving to {out_path}")

    ## Create the enriched mapping
    logging.info("Creating enriched mapping (object and name)...")
    topic_to_info = defaultdict(lambda: {"oggetto": [], "nome": []})

    # Iterate over the data
    for doc, topic, oggetto, nome in zip(texts, topics, meta_df["oggetto"], meta_df["nome"]):
        # No need for 'if doc in doc_to_info', they are guaranteed to be aligned
        topic_to_info[topic]["oggetto"].append(oggetto)
        topic_to_info[topic]["nome"].append(nome)

    # Apply the mapping to the topic info DataFrame
    topic_info_df["Representative_Oggetto"] = topic_info_df["Topic"].map(
        lambda x: ", ".join(topic_to_info.get(x, {}).get("oggetto", []))
    )
    topic_info_df["Representative_Nome"] = topic_info_df["Topic"].a(
        lambda x: ", ".join(topic_to_info.get(x, {}).get("nome", []))
    )

    out_path = os.path.join(output_dir, "topic_info_enriched.csv")
    topic_info_df.to_csv(out_path, index=False)
    logging.info(f"CSV file with topic metadata saved to: {out_path}")

    logging.info("Visualization pipeline completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained BERTopic model.")
    
    parser.add_argument(
        "-r", 
        "--run_dir", 
        type=str, 
        required=True, 
        help="Input directory containing the artifacts saved by 'train_bertopic.py'."
    )
    
    parser.add_argument(
        "-o", 
        "--output_dir", 
        type=str, 
        required=True, 
        help="Output directory where HTML visualizations and CSVs will be saved."
    )
    
    args = parser.parse_args()
    main(args.run_dir, args.output_dir)
