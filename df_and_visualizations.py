# Crea mapping testo > oggetto, nome
doc_to_info = {doc: (oggetto, nome) for doc, oggetto, nome in zip(df["testo"], df["oggetto"], df["nome"])}

# Vocabolario per registrare il mapping
topic_to_info = {}

# Itera sui topic e sui documenti
for doc, topic in zip(df["testo"], topics):
    if topic not in topic_to_info:
        topic_to_info[topic] = {"oggetto": [], "nome": []}

    if doc in doc_to_info:  # Condizione per controllare che il documento sia presente
        topic_to_info[topic]["oggetto"].append(doc_to_info[doc][0])  # Aggiungi oggetto
        topic_to_info[topic]["nome"].append(doc_to_info[doc][1])  # Aggiungi nome

# Converti lista per maggior leggibilità
df_info_reduced["Representative_Oggetto"] = df_info_reduced["Topic"].map(lambda x: ", ".join(topic_to_info.get(x, {}).get("oggetto", [])))
df_info_reduced["Representative_Nome"] = df_info_reduced["Topic"].map(lambda x: ", ".join(topic_to_info.get(x, {}).get("nome", [])))

# Visualizzazione dei topic
#più info sulla visualizzazione dei topic o dei documenti:
#https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html
fig = topic_model_reduced.visualize_topics(custom_labels=True)

# visualizzazione dei documenti (per ottimizzare il processo, riduciamo le dimensioni degli embeddings)
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
#questo ci permette di visualizzare gli oggetti passando sopra con il cursore ai punti
nomi = df["nome"].tolist()
fig_docs = topic_model_reduced.visualize_documents(nomi, reduced_embeddings=reduced_embeddings, custom_labels=True)
fig_docs.show()

#Distribuzione topic per direzione
classes = df['direzione'].tolist()
topics_per_class_reduced = topic_model_reduced.topics_per_class(texts, classes=classes)

topic_model_reduced.visualize_topics_per_class(topics_per_class_reduced, top_n_topics=10, custom_labels=True)
