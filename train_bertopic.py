# -*- coding: utf-8 -*-
"""
Script per il training di un modello BERTopic su testi legali italiani.

Questo script carica un file di testo (CSV o Parquet), calcola gli embeddings
utilizzando un modello Hugging Face e addestra un modello BERTopic.
Il modello addestrato e tutti gli artefatti necessari per la pipeline
(embeddings, probabilità, metadati) vengono salvati in una directory di output.

Esempio di esecuzione:
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

# --- Configurazione del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Costanti e Configurazioni del Modello ---
MODEL_NAME = "dlicari/lsg16k-Italian-Legal-BERT-SC"
EMBEDDING_BATCH_SIZE = 4
EMBEDDING_MAX_LENGTH = 4000 # Specifico per questo modello Longformer

# Configura UMAP 
UMAP_MODEL = UMAP(n_neighbors=15,
                  n_components=10,
                  min_dist=0.01,
                  metric="cosine")

# Configura HDBSCAN
HDBSCAN_MODEL = HDBSCAN(min_cluster_size=20,
                        min_samples=1,
                        cluster_selection_epsilon=0.03,
                        metric='euclidean',
                        prediction_data=True)

# Configura c-TF-IDF
CTFIDF_MODEL = ClassTfidfTransformer(reduce_frequent_words=True)


def load_data(filepath: str, text_column: str = "testo") -> pd.DataFrame:
    """
    Carica i dati da un file CSV o Parquet e li filtra.

    Args:
        filepath: Percorso del file (.csv o .parquet).
        text_column: Nome della colonna contenente i testi.

    Returns:
        Un DataFrame filtrato contenente solo le righe con testi validi.
    """
    logging.info(f"Caricamento dati da {filepath}...")
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        raise ValueError("Formato file non supportato. Usare .csv o .parquet")

    if text_column not in df.columns:
        raise ValueError(f"Colonna '{text_column}' non trovata nel file.")

    # Creiamo una maschera booleana per i testi validi
    # Un testo è valido se è una stringa (non NA/null) e ha più di 1 carattere
    mask = (df[text_column].apply(lambda x: isinstance(x, str) and len(x) > 1))
    
    original_count = len(df)
    filtered_df = df[mask].copy() # .copy() per evitare SettingWithCopyWarning
    new_count = len(filtered_df)
    
    logging.info(f"Caricati {original_count} record.")
    logging.info(f"Rimossi {original_count - new_count} record non validi (nulli, non-stringa o corti).")
    logging.info(f"Elaborazione di {new_count} documenti validi.")
    
    # Resetta l'indice per assicurarsi che sia pulito e contiguo
    filtered_df.reset_index(drop=True, inplace=True) 
    
    return filtered_df


def get_embeddings(texts: list[str], model, tokenizer, device, batch_size: int) -> np.ndarray:
    """
    Calcola gli embeddings per una lista di testi in batch.
    """
    embeddings = []
    model.to(device)
    model.eval()  # Imposta il modello in modalità valutazione (importante!)

    for i in tqdm(range(0, len(texts), batch_size), desc="Calcolo degli embeddings"):
        batch_texts = texts[i:i + batch_size]
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=EMBEDDING_MAX_LENGTH, 
            return_tensors="pt"
        )
        
        # Sposta i tensori sul device corretto
        encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}
        
        with torch.no_grad():  # Disattiva il calcolo del gradiente
            model_output = model(**encoded_input)
        
        # Estrai l'embedding (es. token [CLS]) e spostalo sulla CPU per numpy
        batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # Libera memoria solo se su GPU

    return np.array(embeddings)


def main(input_file: str, output_dir: str):
    """
    Funzione principale per eseguire l'intero processo di topic modeling.
    """
    
    # 1. Imposta il device (GPU se disponibile, altrimenti CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Utilizzo del device: {device}")

    # 2. Carica i dati
    # filtered_df ora contiene TUTTE le colonne (testo, oggetto, nome, ecc.)
    # ma SOLO per le righe con testi validi.
    filtered_df = load_data(input_file, text_column="testo")

    # Estrai la lista di testi dal DataFrame filtrato
    texts = filtered_df["testo"].tolist() 
    
    if not texts:
        logging.error("Nessun testo valido trovato nel file di input. Uscita.")
        return

    # 3. Carica il modello e il tokenizer
    logging.info(f"Caricamento modello: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    # 4. Calcola gli embeddings
    embeddings = get_embeddings(texts, model, tokenizer, device, batch_size=EMBEDDING_BATCH_SIZE)
    logging.info(f"Forma embeddings: {embeddings.shape}")

    # 5. Normalizza gli embeddings
    embeddings = normalize(embeddings)

    # 6. Inizializza BERTopic
    logging.info("Inizializzazione modello BERTopic...")
    topic_model = BERTopic(
        embedding_model=None,  # Forniamo noi gli embeddings pre-calcolati
        calculate_probabilities=True, # FONDAMENTALE per lo script 3
        umap_model=UMAP_MODEL,
        hdbscan_model=HDBSCAN_MODEL,
        ctfidf_model=CTFIDF_MODEL,
        verbose=True,
        top_n_words=5
    )

    # 7. Addestra il modello
    logging.info("Inizio addestramento BERTopic...")
    # Otteniamo sia i topic che le probabilità
    topics, probs = topic_model.fit_transform(texts, embeddings)
    logging.info("Addestramento completato.")

    # 8. Salva tutti gli artefatti
    # Assicurati che la directory di output esista
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 8a. Salva il modello e le info sui topic ---
    
    model_path = os.path.join(output_dir, "bertopic_model")
    topic_model.save(model_path, serialization="safetensors")
    logging.info(f"Modello BERTopic salvato in: {model_path}")

    topic_info = topic_model.get_topic_info()
    csv_path = os.path.join(output_dir, "topic_info.csv")
    topic_info.to_csv(csv_path, index=False)
    logging.info(f"Informazioni sui topic salvate in: {csv_path}")

    # --- 8b. Salva gli artefatti per la pipeline (MODIFICHE CHIAVE) ---
    
    # Salva gli embeddings (allineati)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    logging.info(f"Embeddings salvati in: {os.path.join(output_dir, 'embeddings.npy')}")

    # Salva i topic assegnati (allineati)
    np.save(os.path.join(output_dir, "topics.npy"), topics)
    logging.info(f"Topic assegnati salvati in: {os.path.join(output_dir, 'topics.npy')}")

    # Salva le probabilità (allineate) - NECESSARIO PER SCRIPT 3
    np.save(os.path.join(output_dir, "probabilities.npy"), probs)
    logging.info(f"Probabilità salvate in: {os.path.join(output_dir, 'probabilities.npy')}")
    
    # Salva i metadati filtrati (allineati) - NECESSARIO PER SCRIPT 2 e 3
    # Questo DataFrame contiene 'testo', 'oggetto', 'nome', 'direzione', ecc.
    # ed è perfettamente allineato con embeddings.npy, topics.npy, e probabilities.npy
    metadata_path = os.path.join(output_dir, "metadata.parquet")
    filtered_df.to_parquet(metadata_path, index=False)
    logging.info(f"Metadati allineati salvati in: {metadata_path}")
    
    # --- Fine ---

    logging.info(f"Processo completato. Trovati {len(set(topics)) - 1} topic (escluso -1).")
    print(f"\nTopics trovati: {len(set(topics)) - 1}")


if __name__ == "__main__":
    # Questo blocco viene eseguito solo quando lo script è lanciato direttamente
    
    parser = argparse.ArgumentParser(description="Addestra un modello BERTopic su testi legali.")
    
    parser.add_argument(
        "-i", 
        "--input", 
        type=str, 
        required=True, 
        help="Percorso del file di input (.csv o .parquet) contenente la colonna 'testo'."
    )
    
    parser.add_argument(
        "-o", 
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory dove salvare il modello addestrato e tutti gli artefatti della pipeline."
    )
    
    args = parser.parse_args()
    
    main(args.input, args.output_dir)
