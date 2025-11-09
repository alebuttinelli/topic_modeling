# -*- coding: utf-8 -*-
"""
Script per la visualizzazione di un modello BERTopic addestrato.

Carica un modello, gli embeddings, i topic e i metadati da una
directory di "run" (creata da train_bertopic.py) e genera
visualizzazioni interattive in HTML.

Esempio di esecuzione:
    python visualize_bertopic.py -r ./modelli_addestrati -o ./visualizzazioni
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

# --- Configurazione del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(run_dir: str, output_dir: str):
    """
    Funzione principale per caricare i dati e generare visualizzazioni.
    """
    
    # 1. Verifica e crea la directory di output
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Directory di output: {output_dir}")

    # 2. Definisci i percorsi degli artefatti (basati su run_dir)
    model_path = os.path.join(run_dir, "bertopic_model")
    embeddings_path = os.path.join(run_dir, "embeddings.npy")
    topics_path = os.path.join(run_dir, "topics.npy")
    texts_path = os.path.join(run_dir, "texts.json") # Assumendo che Script 1 salvi questo
    metadata_path = os.path.join(run_dir, "metadata.parquet") # Assumendo che Script 1 salvi questo
    topic_info_path = os.path.join(run_dir, "topic_info.csv")

    # 3. Carica tutti gli artefatti
    logging.info("Caricamento artefatti del modello...")
    try:
        topic_model = BERTopic.load(model_path)
        embeddings = np.load(embeddings_path)
        topics = np.load(topics_path)
        texts = pd.read_json(texts_path).squeeze().tolist() # .squeeze() da Series a lista
        meta_df = pd.read_parquet(metadata_path)
        topic_info_df = pd.read_csv(topic_info_path)
    except FileNotFoundError as e:
        logging.error(f"Errore: File non trovato. {e}")
        logging.error(f"Assicurati che '{run_dir}' contenga tutti gli artefatti di training.")
        return

    logging.info("Tutti gli artefatti caricati con successo.")
    
    # 4. Genera Visualizzazione 1: Visualize Topics
    logging.info("Generazione visualizzazione 'visualize_topics'...")
    fig_topics = topic_model.visualize_topics(custom_labels=True)
    out_path = os.path.join(output_dir, "visualize_topics.html")
    fig_topics.write_html(out_path)
    logging.info(f"Salvataggio in {out_path}")

    # 5. Genera Visualizzazione 2: Visualize Documents
    logging.info("Generazione visualizzazione 'visualize_documents'...")
    logging.info("... (Calcolo ridotto UMAP per la visualizzazione)...")
    
    # Questo è lo stesso del tuo codice, ma ora usa `embeddings` caricati
    reduced_embeddings = UMAP(n_neighbors=10, 
                              n_components=2, 
                              min_dist=0.0, 
                              metric='cosine').fit_transform(embeddings)
    
    nomi = meta_df["nome"].tolist() # Usa i metadati ALLINEATI
    fig_docs = topic_model.visualize_documents(nomi, 
                                               reduced_embeddings=reduced_embeddings, 
                                               custom_labels=True)
    out_path = os.path.join(output_dir, "visualize_documents.html")
    fig_docs.write_html(out_path)
    logging.info(f"Salvataggio in {out_path}")

    # 6. Genera Visualizzazione 3: Topics per Class
    logging.info("Generazione visualizzazione 'visualize_topics_per_class'...")
    classes = meta_df['direzione'].tolist() # Usa i metadati ALLINEATI
    topics_per_class = topic_model.topics_per_class(texts, classes=classes)
    
    fig_class = topic_model.visualize_topics_per_class(topics_per_class, 
                                                       top_n_topics=10, 
                                                       custom_labels=True)
    out_path = os.path.join(output_dir, "visualize_topics_per_class.html")
    fig_class.write_html(out_path)
    logging.info(f"Salvataggio in {out_path}")

    # 7. Crea il mapping arricchito (il tuo codice, reso robusto)
    logging.info("Creazione mapping arricchito (oggetto e nome)...")
    topic_to_info = defaultdict(lambda: {"oggetto": [], "nome": []})

    # Itera sui dati ALLINEATI. Questo è sicuro e corretto.
    for doc, topic, oggetto, nome in zip(texts, topics, meta_df["oggetto"], meta_df["nome"]):
        # Non serve 'if doc in doc_to_info', sono garantiti essere allineati
        topic_to_info[topic]["oggetto"].append(oggetto)
        topic_to_info[topic]["nome"].append(nome)

    # Applica il mapping al DataFrame delle info sui topic
    topic_info_df["Representative_Oggetto"] = topic_info_df["Topic"].map(
        lambda x: ", ".join(topic_to_info.get(x, {}).get("oggetto", []))
    )
    topic_info_df["Representative_Nome"] = topic_info_df["Topic"].map(
        lambda x: ", ".join(topic_to_info.get(x, {}).get("nome", []))
    )

    out_path = os.path.join(output_dir, "topic_info_arricchito.csv")
    topic_info_df.to_csv(out_path, index=False)
    logging.info(f"File CSV con metadati dei topic salvato in: {out_path}")

    logging.info("Pipeline di visualizzazione completata.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizza un modello BERTopic addestrato.")
    
    parser.add_argument(
        "-r", 
        "--run_dir", 
        type=str, 
        required=True, 
        help="Directory di input che contiene gli artefatti salvati da 'train_bertopic.py'."
    )
    
    parser.add_argument(
        "-o", 
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory di output dove salvare le visualizzazioni HTML e i CSV."
    )
    
    args = parser.parse_args()
    main(args.run_dir, args.output_dir)
