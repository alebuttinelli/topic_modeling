##Analisi della distribuzione delle probabilità dei documenti
#creadzione df con distribuzione di probabilità per ogni documento
df_probs = pd.DataFrame(probs)

#Threshold di incertezza
uncertainty_threshold = 0.5

#Individuazione dei documenti al di sotto della soglia di incertezza
uncertain_docs = df_probs[df_probs.max(axis=1) < uncertainty_threshold]

#Indici dei documenti corrispondenti
uncertain_doc_indices = uncertain_docs.index

from scipy.stats import entropy

# Distribuzione di probabilità
uncertain_probs = np.array([probs[idx] for idx in uncertain_doc_indices])

# Calcolo dell'entropia (maggiore entropia maggiore incertezza)
uncertainty_scores = entropy(uncertain_probs.T)

# Colonne per topic
topic_columns = [f"Topic_{i}" for i in range(uncertain_probs.shape[1])]

df_uncertain = pd.DataFrame({
    "idx": uncertain_doc_indices,
    "oggetti": [oggetti[idx] for idx in uncertain_doc_indices],
    "uncertainty": uncertainty_scores
})

df_probs = pd.DataFrame(uncertain_probs, columns=topic_columns)
df_uncertain = pd.concat([df_uncertain, df_probs], axis=1)

# Ordina in modo discendente in base all'incertezza
df_uncertain = df_uncertain.sort_values(by="uncertainty", ascending=False)from scipy.stats import entropy

# Distribuzione di probabilità
uncertain_probs = np.array([probs[idx] for idx in uncertain_doc_indices])

# Calcolo dell'entropia (maggiore entropia maggiore incertezza)
uncertainty_scores = entropy(uncertain_probs.T)

# Colonne per topic
topic_columns = [f"Topic_{i}" for i in range(uncertain_probs.shape[1])]

df_uncertain = pd.DataFrame({
    "idx": uncertain_doc_indices,
    "oggetti": [oggetti[idx] for idx in uncertain_doc_indices],
    "uncertainty": uncertainty_scores
})

df_probs = pd.DataFrame(uncertain_probs, columns=topic_columns)
df_uncertain = pd.concat([df_uncertain, df_probs], axis=1)

# Ordina in modo discendente in base all'incertezza
df_uncertain = df_uncertain.sort_values(by="uncertainty", ascending=False)
