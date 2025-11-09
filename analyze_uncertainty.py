# -*- coding: utf-8 -*-
"""
Script per l'analisi dell'incertezza del modello BERTopic.

Carica la matrice delle probabilità e i metadati da una 'run' di training,
identifica i documenti con un'alta incertezza (bassa probabilità massima
o alta entropia) e salva un CSV con questi documenti e i loro punteggi.

Esempio di esecuzione:
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

# --- Configurazione del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(run_dir: str, output_file: str, threshold: float):
    """
    Funzione principale per analizzare l'incertezza.
    """
    
    # 1. Definisci i percorsi degli artefatti
    probs_path = os.path.join(run_dir, "probabilities.npy")
    metadata_path = os.path.join(run_dir, "metadata.parquet") # Assumiamo che 'oggetto' sia qui

    # 2. Carica gli artefatti necessari
    logging.info("Caricamento artefatti...")
    try:
        probs = np.load(probs_path)
        meta_df = pd.read_parquet(metadata_path)
    except FileNotFoundError as e:
        logging.error(f"Errore: File non trovato. {e}")
        logging.error(f"Assicurati che '{run_dir}' contenga 'probabilities.npy' e 'metadata.parquet'.")
        return
    
    logging.info(f"Caricati {probs.shape[0]} probabilità e {len(meta_df)} record di metadati.")
    
    # 3. Identifica i documenti incerti (basato sul tuo codice)
    # (Ho rimosso il codice duplicato)
    
    # Creazione df con distribuzione di probabilità per ogni documento
    df_probs = pd.DataFrame(probs)

    # Individuazione dei documenti al di sotto della soglia di incertezza
    uncertain_mask = df_probs.max(axis=1) < threshold
    uncertain_doc_indices = df_probs[uncertain_mask].index
    
    logging.info(f"Trovati {len(uncertain_doc_indices)} documenti sotto la soglia di {threshold}.")

    if len(uncertain_doc_indices) == 0:
        logging.warning("Nessun documento incerto trovato con questa soglia. Lo script termina.")
        return

    # 4. Calcola l'entropia (maggiore entropia = maggiore incertezza)
    # Seleziona solo le righe delle probabilità per i documenti incerti
    uncertain_probs_array = probs[uncertain_doc_indices]

    # Calcola l'entropia per ogni riga (trasponi l'array per `entropy`)
    uncertainty_scores = entropy(uncertain_probs_array.T)

    # 5. Prepara i DataFrame per l'output
    
    # Colonne per i topic
    topic_columns = [f"Topic_{i}" for i in range(probs.shape[1])]

    # DataFrame con le informazioni di base
    # (Usiamo .iloc[] per selezionare i metadati corretti e resettiamo l'indice)
    df_uncertain_info = pd.DataFrame({
        "original_index": uncertain_doc_indices,
        "oggetti": meta_df.iloc[uncertain_doc_indices]["oggetto"].values,
        "uncertainty_score (entropy)": uncertainty_scores,
        "max_probability": df_probs.iloc[uncertain_doc_indices].max(axis=1).values
    })

    # DataFrame con le probabilità per ogni topic (solo per i doc incerti)
    df_uncertain_probs = pd.DataFrame(uncertain_probs_array, columns=topic_columns)

    # Unisci i due DataFrame fianco a fianco (ora hanno lo stesso indice 0..N)
    df_final_output = pd.concat([df_uncertain_info, df_uncertain_probs], axis=1)

    # Ordina in modo discendente in base all'incertezza (entropia)
    df_final_output = df_final_output.sort_values(by="uncertainty_score (entropy)", ascending=False)

    # 6. Salva il risultato
    # Assicurati che la directory di output esista
    output_dir = os.path.dirname(output_file)
    if output_dir: # Controlla se il percorso non è vuoto (se l'utente vuole salvare nella root)
        os.makedirs(output_dir, exist_ok=True)
    
    df_final_output.to_csv(output_file, index=False)
    logging.info(f"File CSV con l'analisi di incertezza salvato in: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analizza l'incertezza dei documenti in un modello BERTopic.")
    
    parser.add_argument(
        "-r", 
        "--run_dir", 
        type=str, 
        required=True, 
        help="Directory di input che contiene 'probabilities.npy' e 'metadata.parquet'."
    )
    
    parser.add_argument(
        "-o", 
        "--output_file", 
        type=str, 
        required=True, 
        help="Percorso file .csv dove salvare l'analisi di incertezza."
    )
    
    parser.add_argument(
        "-t", 
        "--threshold", 
        type=float, 
        default=0.5,
        help="Soglia di probabilità massima per considerare un documento 'incerto'. (Default: 0.5)"
    )
    
    args = parser.parse_args()
    main(args.run_dir, args.output_file, args.threshold)
