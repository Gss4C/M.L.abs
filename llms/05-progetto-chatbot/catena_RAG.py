import os
from dotenv import load_dotenv 
from langchain_community.vectorstores import FAISS
#from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
#from getpass import getpass

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Configurazione della Chiave API di OpenAI
# Carichiamo le variabili d'ambiente dal file .env
load_dotenv() 

# Controllo di sicurezza per la chiave API
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ or not os.environ["HUGGINGFACEHUB_API_TOKEN"]:
    print("ERRORE: La variabile d'ambiente HUGGINGFACEHUB_API_TOKEN non è stata impostata o è vuota.")
    print("Assicurati di avere un file .env nella stessa cartella dello script con il contenuto: OPENAI_API_KEY='la-tua-chiave'")
    exit()

print("Chiave API di Groq trovata tramite file .env")

# Creazione del Retriever
# In una vera applicazione, questo verrebbe caricato da un file.
print("\n--- Sto creando la Knowledge Base e il Retriever ---")
knowledge_base_texts = [
    "Il reset della password per l'account utente si effettua tramite il link 'Password dimenticata?' presente nella pagina di login. Una volta cliccato, l'utente dovrà inserire l'indirizzo email associato al proprio account.",
    "Dopo aver inserito l'email, il sistema invierà un messaggio di posta elettronica contenente un link sicuro e valido per 60 minuti. Cliccando su quel link, l'utente potrà impostare una nuova password.",
    "La nuova password deve essere lunga almeno 8 caratteri e contenere almeno una lettera maiuscola, un numero e un simbolo speciale (es. !, @, #, $). Non è possibile riutilizzare una delle ultime 5 password.",
    "L'assistenza clienti è disponibile via chat sul sito dalle 9:00 alle 18:00 dal lunedì al venerdì. Per problemi urgenti fuori orario, è possibile aprire un ticket via email all'indirizzo support@examplecorp.com."
]

documents        = [Document(page_content=text) for text in knowledge_base_texts]
text_splitter    = RecursiveCharacterTextSplitter(
    chunk_size    = 500, 
    chunk_overlap = 50
)
chunks           = text_splitter.split_documents(documents)
embeddings_model = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)
vector_store     = FAISS.from_documents(chunks, embeddings_model)
retriever        = vector_store.as_retriever(search_kwargs={"k": 3})

print("Retriever basato su FAISS è pronto!")

# Definire il Prompt Template
print("\n--- Sto definendo il Prompt Template per RAG... ---")

template_text = """
Sei un assistente AI utile e preciso. Rispondi alla domanda dell'utente basandoti ESCLUSIVAMENTE sul contesto fornito qui sotto.
Il contesto è estratto da documenti specifici di una knowledge base.
Se le informazioni necessarie per rispondere non sono presenti nel contesto, dì esplicitamente "Mi dispiace, non ho trovato informazioni sufficienti nel contesto fornito per rispondere a questa domanda.".
Non aggiungere informazioni non presenti nel contesto. Non inventare risposte.

Contesto:
{context}

Domanda:
{question}

Risposta Utile:
"""

rag_prompt_template = PromptTemplate(
    template        = template_text,
    input_variables = ["context", "question"]
)

print("Prompt Template creato con successo.")

# Inizializzare l'LLM
print("\n--- Sto inizializzando il modello LLM... ---")
'''
llm = ChatHuggingFace(
    model_name   = "deepseek-ai/DeepSeek-R1-0528", 
    temperature  = 0
)
'''
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass("Enter your Hugging Face API key: ")
model = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2.6",
    huggingfacehub_api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"] ,
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

llm = ChatHuggingFace(llm=model)

print(f"Modello LLM ({llm.model_id}) inizializzato.")

# Costruire ed Eseguire la Catena RAG
print("\n--- Sto costruendo la catena RAG con LCEL... ---")
rag_chain = (
    {
        "context": retriever | format_docs, #retriver restituisce lista oggetti, devo unirli
        "question": RunnablePassthrough()   #passo domanda utente senza modifiche
    }
    | rag_prompt_template
    | llm
    | StrOutputParser()
)
print("Catena RAG pronta per l'uso!")

# Test della Catena RAG 
print("\n" + "="*50)
print("INIZIO TEST DELLA CATENA RAG")
print("="*50 + "\n")

# Domanda le cui informazioni sono presenti nella Knowledge Base
user_question_1 = "Come posso resettare la mia password?"

print(f"DOMANDA 1 (in-context): '{user_question_1}'")
final_answer_1 = rag_chain.invoke(user_question_1)
print("\nRISPOSTA FINALE DAL CHATBOT:")
print(final_answer_1)
print("-" * 50)

# Domanda le cui informazioni NON sono presenti nella Knowledge Base
user_question_2 = "Chi ha vinto il festival di Sanremo quest'anno?"

print(f"\nDOMANDA 2 (out-of-context): '{user_question_2}'")
final_answer_2 = rag_chain.invoke(user_question_2)
print("\nRISPOSTA FINALE DAL CHATBOT:")
print(final_answer_2)
print("=" * 50)
print("TEST COMPLETATO.")
print("=" * 50)