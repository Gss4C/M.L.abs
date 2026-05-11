#Come generare embedding con HF-Hub e locale
#come vedere similarità coseno tramite HF

import numpy as np
import torch
import os
from google.colab import userdata
from sentence_transformers import SentenceTransformer
import groq
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    "Il gatto dorme sul divano.",
    "Il felino si riposa sulla poltrona.",
    "Domani pioverà a Roma."
] #frasi di esempio per fare embedding e similarità semantica (cosine similarity)


print("=== ESEMPIO DI APPROCCIO CON CALCOLO LOCALE ===")
st_model_name = 'all-MiniLM-L6-v2'
st_model = SentenceTransformer(st_model_name)


if torch.cuda.is_available():
    print(f"Sentence Transformers sta utilizzando la GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Sentence Transformers sta utilizzando la CPU.")

# Generazione degli embeddings per le nostre frasi
st_embeddings = st_model.encode(sentences)

# Analisi degli embeddings generati
print(f"\nDimensioni della matrice degli embeddings (Sentence Transformers): {st_embeddings.shape}")
# Output atteso: (numero_di_frasi, dimensione_embedding_del_modello)
# Per 'all-MiniLM-L6-v2', la dimensione è 384 -> (3, 384)


print("\n\n=== ESEMPIO DI APPROCCIO CON CHIAMATA A OPENAI ===")
#osare api che si vuole
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
openai_client = openai.OpenAI()

OPENAI_embeddings_np = None

openai_model_name = "text-embedding-ada-002"
print(f"\nRichiesta di embeddings a OpenAI con il modello '{openai_model_name}'...")

response = openai_client.embeddings.create(
    model = openai_model_name,
    input = sentences 
)
# Estraiamo i vettori dalla risposta
openai_embeddings_list = [item.embedding for item in response.data]
openai_embeddings_np = np.array(openai_embeddings_list)
print(f"\nEmbeddings generati con OpenAI ('{openai_model_name}'):")
print(f"Dimensioni della matrice: {openai_embeddings_np.shape}")


print("\n\n === ESEMPIO CON HUGGING FACE ===")

from huggingface_hub import InferenceClient
import numpy as np

client = InferenceClient( api_key = userdata.get('HFT_KEY'))

#hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_model_name = "intfloat/multilingual-e5-large"

hf_embeddings_np = np.array([
    client.feature_extraction(sentence, model=hf_model_name)
    for sentence in sentences
])

print(f"Dimensioni della matrice: {hf_embeddings_np.shape}")
print(f"Primi 5 valori del primo embedding: {hf_embeddings_np[0][:5]}")

print("--- Similarità Coseno con Sentence Transformers ---")
if 'hf_embeddings_np' in locals() and isinstance(hf_embeddings_np, np.ndarray):
    similarity_matrix_st = cosine_similarity(hf_embeddings_np)

    print("\nMatrice di Similarità Coseno (Sentence Transformers - all-MiniLM-L6-v2):")
    print(np.around(similarity_matrix_st, decimals=4))

    print(f"\nInterpretazione (all-MiniLM-L6-v2):")
    print(f"Frase 0: '{sentences[0]}'")
    print(f"Frase 1: '{sentences[1]}'")
    print(f"Frase 2: '{sentences[2]}'")
    print(f"Similarità tra Frase 0 e Frase 1 (gatto vs gatto): {similarity_matrix_st[0][1]:.4f}")
    print(f"Similarità tra Frase 0 e Frase 2 (gatto vs meteo): {similarity_matrix_st[0][2]:.4f}")
    print(f"Similarità tra Frase 1 e Frase 2 (gatto vs meteo): {similarity_matrix_st[1][2]:.4f}")
else:
    print("\nEmbeddings da Sentence Transformers non disponibili per il calcolo della similarità.")

print("\n\n--- Similarità Coseno con OpenAI Embeddings ---")
if openai_embeddings_np is not None and isinstance(openai_embeddings_np, np.ndarray):
    similarity_matrix_openai = cosine_similarity(openai_embeddings_np)

    print("\nMatrice di Similarità Coseno (OpenAI - text-embedding-ada-002):")
    print(np.around(similarity_matrix_openai, decimals=4))

    print(f"\nInterpretazione (text-embedding-ada-002):")
    print(f"Frase 0: '{sentences[0]}'")
    print(f"Frase 1: '{sentences[1]}'")
    print(f"Frase 2: '{sentences[2]}'")
    print(f"Similarità tra Frase 0 e Frase 1 (gatto vs gatto): {similarity_matrix_openai[0][1]:.4f}") # Atteso ~0.9445
    print(f"Similarità tra Frase 0 e Frase 2 (gatto vs meteo): {similarity_matrix_openai[0][2]:.4f}") # Atteso ~0.8026
    print(f"Similarità tra Frase 1 e Frase 2 (gatto vs meteo): {similarity_matrix_openai[1][2]:.4f}") # Atteso ~0.7958
else:
    print("\nEmbeddings da OpenAI non disponibili per il calcolo della similarità.")

