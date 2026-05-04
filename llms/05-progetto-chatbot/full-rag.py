import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS

load_dotenv()
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ or not os.environ["HUGGINGFACEHUB_API_TOKEN"]:
    print("ERRORE: La variabile d'ambiente HUGGINGFACEHUB_API_TOKEN non è stata impostata.")
    exit()

print("\n[PARTE 1] -> Inizio costruzione del modulo di Recupero...")
knowledge_base_texts = [
    "Il reset della password per l'account utente si effettua tramite il link 'Password dimenticata?' presente nella pagina di login. Una volta cliccato, l'utente dovrà inserire l'indirizzo email associato al proprio account.",
    "Dopo aver inserito l'email, il sistema invierà un messaggio di posta elettronica contenente un link sicuro e valido per 60 minuti. Cliccando su quel link, l'utente potrà impostare una nuova password.",
    "La nuova password deve essere lunga almeno 8 caratteri e contenere almeno una lettera maiuscola, un numero e un simbolo speciale (es. !, @, #, $). Non è possibile riutilizzare una delle ultime 5 password.",
    "L'assistenza clienti è disponibile via chat sul sito dalle 9:00 alle 18:00 dal lunedì al venerdì. Per problemi urgenti fuori orario, è possibile aprire un ticket via email all'indirizzo support@examplecorp.com."
]

# Creiamo Document per ciascun testo, assegnando un metadata 'source'
documents = [
    Document(page_content=text, metadata={"source": f"doc_id_{i}"})
    for i, text in enumerate(knowledge_base_texts)
]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"Dati divisi in {len(chunks)} chunk.")

# Creazione del modello di embedding (HuggingFace)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_model = HuggingFaceEmbeddings(
    model_name   = model_name,
    model_kwargs = {'device': 'cpu'}
)
print(f"Modello di embedding '{model_name}' caricato.")

vector_store = FAISS.from_documents(chunks, embeddings_model)
index_file_path = "05-progetto-chatbot/my_faiss_index"
vector_store.save_local(index_file_path)
print(f"Indice FAISS salvato localmente in: '{index_file_path}'")

# Test rapido del retriever
print("\n--- Test del solo Modulo di Recupero ---")
query_test_retriever = "Come posso resettare la mia password?"
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
retrieved_docs = retriever.invoke(query_test_retriever)

print(f"Domanda di test: '{query_test_retriever}'")
print(f"Trovati {len(retrieved_docs)} chunk rilevanti:")
for i, doc in enumerate(retrieved_docs):
    preview = doc.page_content[:80].replace("\n", " ")
    print(f"  - Chunk {i+1} (Fonte: {doc.metadata['source']}): '{preview}...'")

print("[PARTE 1] -> Modulo di Recupero completato e testato!")

print("\n[PARTE 2] -> Inizio costruzione del modulo di Generazione...")
template_text = """
Sei un assistente AI utile e preciso. Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul contesto fornito.
Se le informazioni per rispondere non sono nel contesto, dì: "Mi dispiace, non ho trovato informazioni sufficienti per rispondere.".
Non inventare risposte.

Contesto:
{context}

Domanda:
{question}

Risposta Utile:
"""
rag_prompt_template = PromptTemplate.from_template(template_text)
print("Prompt Template per RAG creato.")

# Inizializzazione del modello LLM (ChatOpenAI)
model = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2.6",
    huggingfacehub_api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"] ,
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  
)

llm = ChatHuggingFace(llm=model)
print(f"Modello LLM ({llm.model_id}) inizializzato.")

# Ricreazione del retriever dal salvataggio (facoltativo)
loaded_vector_store = FAISS.load_local(
    index_file_path,
    embeddings_model,
    allow_dangerous_deserialization=True
)
retriever = loaded_vector_store.as_retriever(search_kwargs={"k": 3})

# Funzione helper per formattare i documenti come stringa di contesto
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Costruzione della catena RAG con LCEL
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt_template
    | llm
    | StrOutputParser()
)
print("Catena RAG completa creata con LCEL.")
print("[PARTE 2] -> Modulo di Generazione completato!")

print("\n" + "="*50 + "\nINIZIO TEST FINALE DELLA PIPELINE RAG\n" + "="*50)

# Test 1: domanda in-context
user_question_1 = "Qual è la procedura per il reset della password e quali sono i requisiti?"
print(f"\nDOMANDA 1 (in-context): '{user_question_1}'")
final_answer_1 = rag_chain.invoke(user_question_1)
print("\nRISPOSTA DAL CHATBOT:")
print(final_answer_1)
print("-" * 50)

# Test 2: domanda out-of-context
user_question_2 = "Qual è il menu del pranzo di oggi?"
print(f"\nDOMANDA 2 (out-of-context): '{user_question_2}'")
final_answer_2 = rag_chain.invoke(user_question_2)
print("\nRISPOSTA DAL CHATBOT:")
print(final_answer_2)

print("\n" + "="*50 + "\nTEST COMPLETATO. PIPELINE FUNZIONANTE!\n" + "="*50)
