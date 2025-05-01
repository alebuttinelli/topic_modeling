# Import delle librerie
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from bertopic import BERTopic
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN

texts = df["testo"].tolist()  # colonna testo come lista di liste
texts = [text for text in texts if type(text) is str and len(text) > 1]

# Caricamento del modello e tokenizer
model_name = "dlicari/lsg16k-Italian-Legal-BERT-SC"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Funzione per estrarre gli embeddings
def get_embeddings(texts, batch_size=4):
    embeddings = []
    model.cuda()
    for i in tqdm(range(0, len(texts), batch_size), desc="Calcolo degli embeddings"):
        batch_texts = texts[i:i + batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=4000, return_tensors="pt")
        encoded_input = {key: tensor.cuda() for key, tensor in encoded_input.items()}  # Trasferisci gli input sulla GPU
        with torch.no_grad():  # Disattiva il calcolo del gradiente
            model_output = model(**encoded_input)  # Elaborazione del modello
        batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()  # Torna alla CPU per numpy
        embeddings.extend(batch_embeddings)
        torch.cuda.empty_cache()  # Libera memoria
    return embeddings

# Estrazione degli embeddings
embeddings = get_embeddings(texts)

## Gestione degli embeddings
# Trasformazione in np.array
embeddings = np.array(embeddings)
# Normalizzazione degli embeddings
embeddings = normalize(embeddings)

###Impostazioni personalizzate del modello HDBSCAN e UMAP per ridurre il numero di doc assegnati a -1
# Configura UMAP con parametri personalizzati
umap_model = UMAP(n_neighbors=15,
                  n_components=10,
                  min_dist=0.01,
                  metric="cosine")
# Configura HDBSCAN con parametri personalizzati
hdbscan_model = HDBSCAN(min_cluster_size=20,
                        min_samples=1,
                        cluster_selection_epsilon=0.03,
                        metric='euclidean',
                        prediction_data=True)

# Creazione della classe cffid per ridurre il peso delle parole più frequenti nella rappresentazione dei topic
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
# Creazione del modello BERTopic
topic_model = BERTopic(embedding_model=None,
                       calculate_probabilities=True,
                       umap_model=umap_model,
                       hdbscan_model=hdbscan_model,
                       ctfidf_model=ctfidf_model,
                       verbose=True,
                       top_n_words=5)

# Adattamento del modello ai dati
topics, probs = topic_model.fit_transform(texts, embeddings)

# Esplorazione dei risultati
print(f"Topics trovati: {len(set(topics)) - 1}")  # Esclude il -1 (outliers)
