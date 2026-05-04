from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("\n--- Preparazione dei documenti e 'chunking'... ---")
knowledge_base_texts = [
    "Il reset della password per l'account utente si effettua tramite il link 'Password dimenticata?' presente nella pagina di login. Una volta cliccato, l'utente dovrà inserire l'indirizzo email associato al proprio account.",
    "Dopo aver inserito l'email, il sistema invierà un messaggio di posta elettronica contenente un link sicuro e valido per 60 minuti. Cliccando su quel link, l'utente potrà impostare una nuova password.",
    "La nuova password deve essere lunga almeno 8 caratteri e contenere almeno una lettera maiuscola, un numero e un simbolo speciale (es. !, @, #, $). Non è possibile riutilizzare una delle ultime 5 password.",
    "L'assistenza clienti è disponibile via chat sul sito dalle 9:00 alle 18:00 dal lunedì al venerdì. Per problemi urgenti fuori orario, è possibile aprire un ticket via email all'indirizzo support@examplecorp.com."
]
documents = [Document(
    page_content = text, 
    metadata     = {"source": f"doc_id_{i}"}) for i, text in enumerate(knowledge_base_texts)]
print(f"Creati {len(documents)} documenti in memoria.")

# Suddivisione in chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size      = 500,
    chunk_overlap   = 50,  #overlap per contesto
    length_function = len,
)
chunks = text_splitter.split_documents(documents)
print(f"Documenti divisi in {len(chunks)} chunk.")


# Generazione degli embeddings
print("\n--- Inizializzazione del modello di embedding... ---")
model_name = "sentence-transformers/all-MiniLM-L6-v2" #leggero, veloce, free
embeddings_model = HuggingFaceEmbeddings(
    model_name   = model_name,
    model_kwargs = {'device': 'cpu'} #alt: cuda
)
print(f"Modello di embedding '{model_name}' caricato.")

# Creazione e salvataggio dell'indice FAISS
print("\n--- Creazione e salvataggio dell'indice vettoriale FAISS... ---")
vector_store = FAISS.from_documents(chunks, embeddings_model)
print("Indice FAISS creato con successo in memoria.")

index_file_path = "05-progetto-chatbot/my_faiss_index"
vector_store.save_local(index_file_path)
print(f"Indice salvato localmente nella cartella: '{index_file_path}'")
#salvato su disco così posso leggerlo in futuro anziché ricrearlo da capo

####################################
# Caricamento e utilizzo dell'indice

print("\n--- Caricamento dell'indice e ricerca semantica... ---")
# Per caricare l'indice FAISS è necessario specificare allow_dangerous_deserialization=True
# per motivi di sicurezza, perché i file pickle possono contenere codice irregolare.
loaded_vector_store = FAISS.load_local(
    index_file_path,
    embeddings_model,
    allow_dangerous_deserialization = True
)
print("Indice FAISS caricato dal disco.")

query = "Come posso resettare la mia password?"
k = 3

print(f"\nEseguo la ricerca per la domanda: '{query}'")
results = loaded_vector_store.similarity_search(query, k=k)

print(f"\n--- Trovati {len(results)} chunk rilevanti: ---")
for i, doc in enumerate(results):
    print(f"\n[ Risultato {i+1} ]")
    print(f"Fonte: {doc.metadata.get('source', 'N/D')}")
    print(f"Contenuto del chunk:\n---\n{doc.page_content}\n---")

print("\n--- Il nostro modulo di recupero è pronto. ---")