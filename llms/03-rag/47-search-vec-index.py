import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from dotenv import load_dotenv
    import pickle
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import torch #Utilizzato per verificare la disponibilità della GPU

    return SentenceTransformer, cosine_similarity, os, pickle


@app.cell
def _():
    index_filepath = "03-rag/my_simple_corpus_index.pkl"
    #loaded_index = None
    return (index_filepath,)


@app.cell
def _(index_filepath, os, pickle):
    print(f"Tentativo di caricare l'indice vettoriale da '{index_filepath}'...")

    if os.path.exists(index_filepath):
        with open(index_filepath, "rb") as f_in:
            loaded_index = pickle.load(f_in)

        if loaded_index:
            print(f"Indice caricato con successo da '{index_filepath}'.")
            print(f"Contiene {len(loaded_index)} elementi (documenti).")
        else:
            print(f"ATTENZIONE: Indice caricato da '{index_filepath}' risulta vuoto o corrotto.")
    else:
        print(f"ERRORE CRITICO: File indice '{index_filepath}' NON TROVATO.")
    return (loaded_index,)


@app.cell
def _(SentenceTransformer, corpus_model):
    model_name_used_for_index = 'paraphrase-multilingual-mpnet-base-v2'
    print(f"Tentativo di gestire il modello '{model_name_used_for_index}'...")

    # Gestione del riutilizzo del modello 'corpus_model'
    # Questa parte assume che se 'corpus_model' esiste e stai usando lo stesso
    # 'model_name_used_for_index', allora possiamo riutilizzarlo.
    reused_existing_model = False
    if 'corpus_model' in locals() and isinstance(corpus_model, SentenceTransformer):
        # Se 'corpus_model' esiste, si assume che sia stato creato con il nome corretto.
        print(f"INFO: Trovata variabile 'corpus_model' di tipo SentenceTransformer.")
        print(f"      Si tenterà di riutilizzarla, assumendo sia il modello '{model_name_used_for_index}'.")
        print(f"      Se hai cambiato 'model_name_used_for_index' da quando 'corpus_model' è stato creato,")
        print(f"      o se 'corpus_model' non è il modello corretto, considera di non eseguire questa parte o di eliminare 'corpus_model'.")
        model = corpus_model
        reused_existing_model = True
        print(f"Riutilizzo del modello da 'corpus_model'. Nome atteso: '{model_name_used_for_index}'.")

    if not reused_existing_model:
        print(f"Caricamento del modello '{model_name_used_for_index}' da zero...")
        model = SentenceTransformer(model_name_used_for_index)
        print(f"Modello '{model_name_used_for_index}' inizializzato.")
    return (model,)


@app.cell
def _(loaded_index, model):
    if loaded_index and model is not None:
        print("Indice vettoriale e modello di embedding sono stati caricati correttamente.")
        print("--------------------------------------------------------------------")
    else:
        print("ERRORE: Impossibile procedere. Indice o modello non validi/caricati.")
        print("--------------------------------------------------------------------")
    return


@app.cell
def _(model):
    # query di esempio
    query = "Quali animali sono considerati i migliori amici dell'uomo?"

    # Generiamo l'embedding vettoriale per la query
    # Importante passare la query come LISTA di una singola stringa al metodo encode(),
    # in modo che l'output sia una matrice 2D (1 riga, N dimensioni) compatibile con sklearn

    query_embedding = model.encode([query])
    print(f"Embedding della query generato. Shape: {query_embedding.shape}")
    # L'output atteso è una tupla
    print(f"Primi 3 valori dell'embedding della query: {query_embedding[0][:3]}")
    return query, query_embedding


@app.cell
def _(cosine_similarity, loaded_index, query_embedding):
    all_similarities = []
    #tecnica brute force, iteriamo su tutti gli elementi dell'indice per trovare quello con la similarità semantica più alta

    print(f"Avvio del calcolo delle similarità tra la query e i {len(loaded_index)} documenti nell'indice...")

    for _i, (doc_embedding, doc_reference) in enumerate(loaded_index):
        # print controllo
        if (_i + 1) % 5 == 0 or _i == 0 or (_i + 1) == len(loaded_index):
            print(f"  Elaborazione documento {_i+1}/{len(loaded_index)}...")

        # 'doc_embedding' è un vettore 1D (shape: dimensione_embedding,)
        # 'query_embedding' è una matrice 2D (shape: 1, dimensione_embedding)

        # Per usare cosine_similarity di scikit-learn, entrambi gli input dovrebbero essere 2D.
        # quindi riformattiamo doc_embedding da (dimensione,) a (1, dimensione).
        current_doc_embedding_reshaped = doc_embedding.reshape(1, -1)

        similarity_score_array = cosine_similarity(query_embedding, current_doc_embedding_reshaped)

        # Estraiamo il valore scalare dal risultato
        score = similarity_score_array[0][0]

        # Aggiungiamo alla lista il punteggio e il riferimento completo al documento
        all_similarities.append( (score, doc_reference) )

    print(f"Calcolo delle similarità completato per tutti i {len(all_similarities)} documenti.")

    return (all_similarities,)


@app.cell
def _(all_similarities, query):
    # Ordiniamo la lista 'all_similarities' in base al punteggio
    # in ordine decrescente (i più simili per primi, quindi reverse=True)
    all_similarities.sort(
        key=lambda item: item[0], 
        reverse=True
    )
    print(f"\n--- Risultati della Ricerca Semantica per la query: '{query}' ---")

    # Definiamo quanti risultati vogliamo mostrare
    num_results_to_show = 3
    print(f"Top {num_results_to_show} documenti più simili (su {len(all_similarities)} totali):")

    # Itera sui primi N risultati ordinati e stampali
    for i, (_score, ref) in enumerate(all_similarities[:num_results_to_show]):
        print(f"\n{i+1}. Punteggio Similarità: {_score:.4f}")
        print(f"   ID Documento: {ref['id']}")
        print(f"   Testo Documento: {ref['text']}")
        print("-" * 40)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
