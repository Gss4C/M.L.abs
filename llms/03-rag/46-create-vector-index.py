import numpy as np
import torch
import pickle  # Per salvare e caricare l'indice
import os  # Per controllare l'esistenza dei file
from sentence_transformers import SentenceTransformer  # Per costruire l'indice

documents = [
    "Il gatto è un animale domestico popolare.",
    "I cani sono noti per la loro lealtà verso i padroni.",
    "Il ciclo di vita di una farfalla include quattro stadi: uovo, larva, pupa e adulto.",
    "Python è un linguaggio di programmazione versatile e ampiamente utilizzato.",
    "L'intelligenza artificiale sta trasformando molti settori industriali.",
    "La ricetta della torta di mele richiede farina, zucchero, burro e mele.",
    "Il sistema solare è composto da otto pianeti che orbitano attorno al Sole.",
    "Imparare a suonare la chitarra richiede pratica costante.",
]

# come trovo i doc più adatti semanticamente ad una richiesta?
# costruiamo indice vettoriale, utile per il RAG
model_name_to_load = "paraphrase-multilingual-mpnet-base-v2"
corpus_model = None
current_model_name_in_memory = None

corpus_model = SentenceTransformer(model_name_to_load)
current_model_name_in_memory = (
    model_name_to_load  # Memorizza il nome del modello caricato
)
print(f"Modello '{current_model_name_in_memory}' caricato con successo.")

# Verifica utilizzo CPU/GPU solo se il modello è stato caricato correttamente

if torch.cuda.is_available():
    try:
        # Tentativo di determinare il device del modello
        device = next(corpus_model.parameters()).device
        if "cuda" in str(device):
            print(f"Il modello per il corpus ('{current_model_name_in_memory}') sta utilizzando la GPU: {torch.cuda.get_device_name(0)} (device: {device})")
        else:
            print(f"Il modello per il corpus ('{current_model_name_in_memory}') è su CPU (device: {device}), nonostante una GPU sia disponibile.")
    except Exception as e:
        print(f"Modello '{current_model_name_in_memory}' caricato. Impossibile determinare il device specifico (errore: {e}).")
        print(f"PyTorch rileva una GPU. SentenceTransformer di solito la usa automaticamente.")
else:
    print(f"Il modello per il corpus ('{current_model_name_in_memory}') sta utilizzando la CPU (nessuna GPU rilevata da PyTorch).")

corpus_embeddings = None

print(f"\nGenerazione embeddings per i {len(documents)} documenti con '{current_model_name_in_memory}'...")
corpus_embeddings = corpus_model.encode(
    documents, 
    show_progress_bar=True
)

print(f"\nEmbeddings generati per tutti i documenti.")
print(f"Shape della matrice degli embeddings: {corpus_embeddings.shape}")
# Output atteso: (numero_documenti, dimensione_embedding)
# Esempio per 'paraphrase-multilingual-mpnet-base-v2': (8, 768)
# Esempio per 'all-MiniLM-L6-v2': (8, 384)


# ora dobbiamo associare i vettori embedding ai doc originali
# usiamo la struttura dati di una lista di tuple: (vettore embedding del doc, rif del doc originale)
simple_index = []
for i, embedding_vector in enumerate(corpus_embeddings):
    # Creiamo un riferimento al documento.
    doc_reference = {
        "id": i,  # Indice del documento nella lista originale
        "text": documents[i],  # Conserviamo il testo completo per ora
        "preview": documents[i][:80] + ("..." if len(documents[i]) > 80 else ""),
    }

    # Aggiungiamo la tupla alla nostra lista indice
    simple_index.append((embedding_vector, doc_reference))

print(f"\nIndice semplice costruito con {len(simple_index)} elementi.")

# Esaminiamo la struttura del primo elemento dell'indice per capire come è fatto
if simple_index:
    print("\nStruttura del primo elemento dell'indice (documento 0):")
    first_item_embedding, first_item_reference = simple_index[0]
    print(f"  - Tipo dell'embedding: {type(first_item_embedding)}")
    print(f"  - Shape dell'embedding: {first_item_embedding.shape}")
    print(f"  - Primi 3 valori dell'embedding: {first_item_embedding[:3]}")
    print(f"  - Tipo del riferimento: {type(first_item_reference)}")
    print(f"  - Contenuto del riferimento: {first_item_reference}")

# Salvataggio indice per evitare di dover rifare sempre l'indice da capo
index_filepath = "my_simple_corpus_index.pkl"
with open(index_filepath, "wb") as f_out:  # 'wb' sta per 'write binary'
    pickle.dump(
        simple_index, 
        f_out, 
        protocol = pickle.HIGHEST_PROTOCOL)
print(f"\nIndice ({len(simple_index)} elementi) salvato con successo nel file: '{index_filepath}'")
print(f"Dimensione del file: {os.path.getsize(index_filepath) / 1024:.2f} KB")


# Ora simuliamo di chiudere e riaprire il programma.
# Caricamento indice
loaded_index = None

with open(index_filepath, "rb") as f_in: #read-binary
    loaded_index = pickle.load(f_in)

if loaded_index:
    print(f"\nIndice caricato con successo da '{index_filepath}'.")
    print(f"L'indice caricato contiene {len(loaded_index)} elementi.")

    '''
    print("\nVerifica struttura del primo elemento dell'indice caricato:")
    first_loaded_embedding, first_loaded_reference = loaded_index[0]
    print(f"  - Tipo dell'embedding: {type(first_loaded_embedding)}")
    print(f"  - Shape dell'embedding: {first_loaded_embedding.shape}")
    print(f"  - Primi 3 valori dell'embedding: {first_loaded_embedding[:3]}")
    print(f"  - Tipo del riferimento: {type(first_loaded_reference)}")
    print(f"  - Contenuto del riferimento: {first_loaded_reference}")
    '''
else:
    print(f"Errore: Il caricamento da '{index_filepath}' non ha prodotto dati o il file è corrotto.")