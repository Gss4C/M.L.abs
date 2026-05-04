import numpy as np
import torch
import os 
import pickle # Per salvare e caricare l'indice
from sentence_transformers import SentenceTransformer # Per costruire l'indice

# Il nostro corpus di documenti
# le tecniche per estrarre conoscenze sono varie
# limitiamoci ad avere una lista scema
documents = [
    "Il gatto è un animale domestico popolare.",
    "I cani sono noti per la loro lealtà verso i padroni.",
    "Il ciclo di vita di una farfalla include quattro stadi: uovo, larva, pupa e adulto.",
    "Python è un linguaggio di programmazione versatile e ampiamente utilizzato.",
    "L'intelligenza artificiale sta trasformando molti settori industriali.",
    "La ricetta della torta di mele richiede farina, zucchero, burro e mele.",
    "Il sistema solare è composto da otto pianeti che orbitano attorno al Sole.",
    "Imparare a suonare la chitarra richiede pratica costante."
]

''' Mostra sample documenti
print(f"Corpus di {len(documents)} documenti di esempio caricato.\n")
for i, doc in enumerate(documents):
    print(f"Doc {i}: {doc[:50]}...") 
'''

# Scelta modello embedding per vettorializzare corpus
model_name_to_load = 'paraphrase-multilingual-mpnet-base-v2'
corpus_model       = None
current_model_name_in_memory = None


print(f"\nTentativo di caricamento del modello: '{model_name_to_load}'...")
try:
    corpus_model = SentenceTransformer(model_name_to_load)
    current_model_name_in_memory = model_name_to_load # Memorizza il nome del modello caricato
    print(f"Modello '{current_model_name_in_memory}' caricato con successo.")
except Exception as e:
    print(f"ERRORE durante il caricamento del modello '{model_name_to_load}': {e}")
    corpus_model = None

# Verifica utilizzo CPU/GPU solo se il modello è stato caricato correttamente
if torch.cuda.is_available():
    device = next(corpus_model.parameters()).device 
    if 'cuda' in str(device):
        print(f"Il modello per il corpus ('{current_model_name_in_memory}') sta utilizzando la GPU: {torch.cuda.get_device_name(0)} (device: {device})")
    else:
        print(f"Il modello per il corpus ('{current_model_name_in_memory}') è su CPU (device: {device}), nonostante una GPU sia disponibile.")
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
if corpus_embeddings is not None: #print controllo
    print(f"Dimensione di un singolo embedding: {corpus_embeddings[0].shape[0]}")
    print(f"Primi 3 valori del primo documento (Doc 0): {corpus_embeddings[0][:3]}")


# Iteriamo sugli embeddings e sui documenti originali contemporaneamente
simple_index = []
for i, embedding_vector in enumerate(corpus_embeddings):
    # Creiamo un riferimento al documento.
    doc_reference = {
        "id": i, 
        "text": documents[i], # Conserviamo il testo completo 
        "preview": documents[i][:80] + ("..." if len(documents[i]) > 80 else "") #utile per verifica
    }

    # Tuplla con embedd e dict con doc ref
    simple_index.append( (embedding_vector, doc_reference) )

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

# Salvataggio index
index_filepath = "my_simple_corpus_index.pkl"
with open(index_filepath, "wb") as f_out: #mode write-binary
    pickle.dump(
        simple_index, 
        f_out, 
        protocol = pickle.HIGHEST_PROTOCOL
    )
print(f"\nIndice ({len(simple_index)} elementi) salvato con successo nel file: '{index_filepath}'")
print(f"Dimensione del file: {os.path.getsize(index_filepath) / 1024:.2f} KB")

#test lettura indice
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
