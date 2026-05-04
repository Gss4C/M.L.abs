import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
import gradio as gr

load_dotenv()
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ or not os.environ["HUGGINGFACEHUB_API_TOKEN"]:
    print("ERRORE: La variabile d'ambiente HUGGINGFACEHUB_API_TOKEN non è stata impostata.")
    exit()

print('Preparazione logica RAG:')
# carica indice e modello embedding
FAISS_INDEX_PATH = '05-progetto-chatbot/my_faiss_index'
if not os.path.exists(FAISS_INDEX_PATH):
    print(f'    Errore: Indice faiss non trovato in {FAISS_INDEX_PATH}')
    exit()

model_name = "sentence-transformers/all-MiniLM-L6-v2"
try:
    embeddings_model = HuggingFaceEmbeddings(
        model_name   = model_name,
        model_kwargs = {'device': 'cpu'}
    )
    print(f"  - Modello di embedding '{model_name}' caricato.")
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings_model,
        allow_dangerous_deserialization = True
    )
    print(f"  - Indice FAISS recuperato")
    print(f'  - Creo retriver e procedo ad istanziare llm')
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    model = HuggingFaceEndpoint(
        repo_id            = "meta-llama/Llama-3.3-70B-Instruct",
        task               = "text-generation",
        max_new_tokens     = 512,
        do_sample          = False,
        repetition_penalty = 1.03,
        provider           = "auto",  
        temperature        = 0.1,
        huggingfacehub_api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"] 
    )
    llm = ChatHuggingFace(llm=model)
    print(f"  - Modello LLM {llm.model_id} inizializzato")
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
    print("  - Prompt Template per RAG creato")

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )
    print("  - RAG Chain creata con successo")
    print('  - Logica RAG inizializzata con successo')
except Exception as e:
    prnt(f"  - Errore nell'inizializzazione della logica RAG: {e}")
    exit()

print('Definizione interfaccia: ')

def get_chatbot_response(message, history):
    """
    Funzione che gradio chiamerà per ogni input utente
    """
    print(f"Input utente: {message}")
    try:
        response = rag_chain.invoke(message)
        print(f"Risposta RAG: {response}")
        return response
    except Exception as e:
        print(f"Errore nell'INVOKE della chain: {e}")
        return "Qualcosa è andato storto nel backend. Riprova."
        
print("  - Configurazione interfaccia utente")
chat_ui = gr.ChatInterface(
    fn          = get_chatbot_response,
    title       = "Assistente virtuale RAG",
    description = "Fai una domanda sulla knowledge base",
    examples    = [
        "Come posso resettare la mia password?",
        "Quali sono i requisiti per una nuova password?"
    ] ,
    cache_examples = False,
    #theme          = "soft"
)
print("  - Configurazine avvenuta con successo")
print("  - Avvio applicazione web con Gradio...\n        Apri l'URL che compare")

chat_ui.launch(share=True)