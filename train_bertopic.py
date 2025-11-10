# -*- coding: utf-8 -*-
"""
Script for training a BERTopic model on Italian legal texts.

This script loads a text file (CSV or Parquet), calculates embeddings
using a Hugging Face model, and trains a BERTopic model.
The trained model and all necessary artifacts for the pipeline
(embeddings, probabilities, metadata) are saved to an output directory.

Example run:
    python train_bertopic.py -i ./dati/miei_testi.csv -o ./modelli_addestrati
"""

import pandas as pd
import torch
import argparse
import os
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModel
from bertopic import BERTopic
from tqdm import tqdm
from sklearn.preprocessing import normalize
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants and Model Configurations
MODEL_NAME = "dlicari/lsg16k-Italian-Legal-BERT-SC"
EMBEDDING_BATCH_SIZE = 4
EMBEDDING_MAX_LENGTH = 4000 # Specific to this Longformer model

# Configure UMAP 
UMAP_MODEL = UMAP(n_neighbors=15,
                  n_components=10,
                  min_dist=0.01,
                  metric="cosine")

# Configure HDBSCAN
HDBSCAN_MODEL = HDBSCAN(min_cluster_size=20,
                        min_samples=1,
                        cluster_selection_epsilon=0.03,
                        metric='euclidean',
                        prediction_data=True)

# Configure c-TF-IDF
CTFIDF_MODEL = ClassTfidfTransformer(reduce_frequent_words=True)


def load_data(filepath, text_column = "testo"):
    """
    Loads data from a CSV or Parquet file and filters it.

    Args:
        filepath: Path to the file (.csv or .parquet).
        text_column: Name of the column containing the texts.

    Returns:
        A filtered DataFrame containing only rows with valid texts.
    """
    logging.info(f"Loading data from {filepath}...")
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the file.")

    # Create a boolean mask for valid texts
    # A text is valid if it is a string (not NA/null) and has more than 1 character
    mask = (df[text_column].apply(lambda x: isinstance(x, str) and len(x) > 1))
    
    original_count = len(df)
    filtered_df = df[mask].copy() # .copy() to avoid SettingWithCopyWarning
    new_count = len(filtered_df)
    
    logging.info(f"Loaded {original_count} records.")
    logging.info(f"Removed {original_count - new_count} invalid records (null, non-string, or short).")
    logging.info(f"Processing {new_count} valid documents.")
    
    # Reset the index to ensure it is clean and contiguous
    filtered_df.reset_index(drop=True, inplace=True) 
    
    return filtered_df


def get_embeddings(texts: list[str], model, tokenizer, device, batch_size: int) -> np.ndarray:
    """
    Calculates embeddings for a list of texts in batches.
    """
    embeddings = []
    model.to(device)
    model.eval()  # Set the model to evaluation mode (important!)

    for i in tqdm(range(0, len(texts), batch_size), desc="Calculating embeddings"):
        batch_texts = texts[i:i + batch_size]
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=EMBEDDING_MAX_LENGTH, 
            return_tensors="pt"
        )
        
        # Move tensors to the correct device
        encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}
        
        with torch.no_grad():  # Disable gradient calculation
            model_output = model(**encoded_input)
        
        # Extract the embedding (e.g., [CLS] token) and move it to the CPU for numpy
        batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # Free memory only if on GPU

    return np.array(embeddings)


def main(input_file: str, output_dir: str):
    """
    Main function to run the entire topic modelling process.
    """
    
    ## Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    ## Load data
    filtered_df = load_data(input_file, text_column="testo")

    # Extract the list of texts from the filtered DataFrame
    texts = filtered_df["testo"].tolist() 
    
    if not texts:
        logging.error("No valid texts found in the input file. Exiting.")
        return

    ## Load the model and tokenizer
    logging.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    ## Calculate embeddings
    embeddings = get_embeddings(texts, model, tokenizer, device, batch_size=EMBEDDING_BATCH_SIZE)
    logging.info(f"Embeddings shape: {embeddings.shape}")

    ## Normalize embeddings
    embeddings = normalize(embeddings)

    ## Initialize BERTopic
    logging.info("Initializing BERTopic model...")
    topic_model = BERTopic(
        embedding_model=None,  
        calculate_probabilities=True,
        umap_model=UMAP_MODEL,
        hdbscan_model=HDBSCAN_MODEL,
        ctfidf_model=CTFIDF_MODEL,
        verbose=True,
        top_n_words=5
    )

    ## Train the model
    logging.info("Starting BERTopic training...")
    topics, probs = topic_model.fit_transform(texts, embeddings)
    logging.info("Training completed.")

    ## Save all artifacts
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    ## Save the model and topic info
    
    model_path = os.path.join(output_dir, "bertopic_model")
    topic_model.save(model_path, serialization="safetensors")
    logging.info(f"BERTopic model saved in: {model_path}")

    topic_info = topic_model.get_topic_info()
    csv_path = os.path.join(output_dir, "topic_info.csv")
    topic_info.to_csv(csv_path, index=False)
    logging.info(f"Topic information saved in: {csv_path}")
    
    # Save embeddings
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    logging.info(f"Embeddings saved in: {os.path.join(output_dir, 'embeddings.npy')}")

    # Save assigned topics
    np.save(os.path.join(output_dir, "topics.npy"), topics)
    logging.info(f"Assigned topics saved in: {os.path.join(output_dir, 'topics.npy')}")

    # Save probabilities
    np.save(os.path.join(output_dir, "probabilities.npy"), probs)
    logging.info(f"Probabilities saved in: {os.path.join(output_dir, 'probabilities.npy')}")
    
    # Save filtered metadata
    metadata_path = os.path.join(output_dir, "metadata.parquet")
    filtered_df.to_parquet(metadata_path, index=False)
    logging.info(f"Aligned metadata saved in: {metadata_path}")

    logging.info(f"Process completed. Found {len(set(topics)) - 1} topics (excluding -1).")
    print(f"\nTopics found: {len(set(topics)) - 1}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train a BERTopic model on legal texts.")
    
    parser.add_argument(
        "-i", 
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input file (.csv or .parquet) containing the 'testo' column."
    )
    
    parser.add_argument(
        "-o", 
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory where the trained model and all pipeline artifacts will be saved."
    )
    
    args = parser.parse_args()
    
    main(args.input, args.output_dir)
