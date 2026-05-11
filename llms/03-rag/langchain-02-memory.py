import os
from google.colab import userdata
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)  # Per la gestione dello storico messaggi
from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)  # Per integrare la memoria nella chain LCEL

try:
    groq_apikey = userdata.get("GROQ_API_KEY")
    if groq_apikey:
        os.environ["GROQ_API_KEY"] = groq_apikey
        print("Token API Groq caricato dai secrets di Colab.")
    else:
        print("api key non trovata nei secrets di Colab.")
except Exception as e:
    print(f"Errore: {e}")

# def modello
llm = ChatGroq(
    model       = "llama-3.3-70b-versatile",
    temperature = 0.7,
)
print(f"Inizializzato modello: {llm.model_name}")

# prompting
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sei un assistente AI utile, rispondi in modo conciso.",
        ),
        MessagesPlaceholder(
            variable_name = "history"
        ),  # Qui LangChain metterà la cronologia dei messaggi
        ("human", "{input}"),  # L'input corrente dell'utente
    ]
)
print("ChatPromptTemplate con MessagesPlaceholder ('history') creato.\n")

# mi serve posto dove salvare cronologia conversazione
# vogliamo creare una funzione che restituisca un oggetto capace di gestire momria per un certo memory id
# Ogni session_id avrà la sua istanza di InMemoryChatMessageHistory.
# Questo 'store' manterrà gli oggetti InMemoryChatMessageHistory.
store = {}  # tiene traccia di tutte le cronologie di ogni sessione

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# costruiamo LCEL e integraimo con memoria
print("--- Composizione Chain LCEL con Memoria ---")
base_chain = prompt | llm

# Avvolgiamo la chain base con RunnableWithMessageHistory
# Questo componente gestirà il caricamento e il salvataggio dei messaggi
# usando la funzione get_session_history che abbiamo definito.
conv_chain_with_history = RunnableWithMessageHistory(
    runnable             = base_chain, 
    get_session_history  = get_session_history,  # La funzione factory per ottenere lo storico dei messaggi per sessione
    input_messages_key   = "input",    # La chiave nel dizionario di input che contiene il messaggio dell'utente
    history_messages_key = "history",  # La chiave nel prompt (MessagesPlaceholder) dove inserire la storia
)
print(
    "Conversation chain con memoria (RunnableWithMessageHistory + InMemoryChatMessageHistory) pronta."
)

print("--- Fine Composizione Chain ---\n")
print("--- Conversazione di Prova ---")
# Scegliamo un ID di sessione univoco per questa conversazione.
# In un'applicazione reale questo ID deve essre dinamico: nome utente, coockie ecc
session_id_attuale = "chat_001"

print(f"--- Inizio Conversazione (Session ID: {session_id_attuale}) ---")

# Primo input
user_input_1 = (
    "Ciao, sono uno studente di intelligenza artificiale e mi chiamo Jon."
)
print(f"Tu (Sessione: {session_id_attuale}): {user_input_1}")
ai_response_1 = conv_chain_with_history.invoke(  # si fa invoke della chain racchiusa nella RWMH
    {
        "input": user_input_1
    },  # L'input dell'utente corrisponde a 'input_messages_key', key impostata prima
    config = {
        "configurable": {"session_id": session_id_attuale}
    }, 
)
# L'output di una chain che termina con un ChatModel è un oggetto AIMessage (o simile).
# Accediamo al contenuto testuale con .content
print(
    f"AI: {ai_response_1.content}"
)

print("\n-----\n")

# Secondo input, test diretto della memoria
user_input_2 = "Come mi chiamo?"
print(f"Tu (Sessione: {session_id_attuale}): {user_input_2}")
try:
    ai_response_2 = conv_chain_with_history.invoke(
        {"input": user_input_2},
        config={"configurable": {"session_id": session_id_attuale}},
    )
    print(f"AI: {ai_response_2.content}")  # Dovrebbe rispondere "Alessio"
except Exception as e:
    print(f"ERRORE nell'invocare la chain (input 2): {e}")

print("\n-----\n")

# Terzo input, ricorda l'argomento generale?
user_input_3 = "Di cosa ti ho parlato riguardo ai miei studi?"
print(f"Tu (Sessione: {session_id_attuale}): {user_input_3}")
try:
    ai_response_3 = conv_chain_with_history.invoke(
        {"input": user_input_3},
        config={"configurable": {"session_id": session_id_attuale}},
    )
    print(
        f"AI: {ai_response_3.content}"
    )  # Dovrebbe citare l'intelligenza artificiale
except Exception as e:
    print(f"ERRORE nell'invocare la chain (input 3): {e}")

print(f"\n--- Fine Conversazione (Session ID: {session_id_attuale}) ---")

print("--- Fine Conversazione di Prova ---\n")

print("--- Ispezione Memoria (dallo store) ---")
# Ispezioniamo direttamente lo 'store' e l'oggetto InMemoryChatMessageHistory
# per la session_id che abbiamo usato nella conversazione.

if "session_id_attuale" in locals() and session_id_attuale in store:
    print(f"\n--- Contenuto della memoria per Session ID: {session_id_attuale} ---")
    # Recupera l'oggetto InMemoryChatMessageHistory specifico per la sessione dallo store
    specific_session_history = store[session_id_attuale]

    if (
        not specific_session_history.messages
    ):  # L'attributo '.messages' contiene la lista dei messaggi
        print(
            "Il buffer di memoria per questa sessione è vuoto. (Hai eseguito la conversazione?)"
        )
    else:
        print(
            f"Numero di messaggi nello storico per '{session_id_attuale}': {len(specific_session_history.messages)}"
        )
        for (
            m
        ) in specific_session_history.messages:  # Iteriamo sulla lista di BaseMessage
            role_emoji = (
                "👤" if m.type == "human" else "🤖"
            )  # Qui mi sembra carino identificare chi dice cosa
            print(f"{role_emoji} ({m.type}): {m.content}")
else:
    print(
        f"Cronologia non trovata per Session ID: {locals().get('session_id_attuale', 'NON DEFINITO')} nello store."
    )
    print(
        f"Lo store attualmente contiene cronologie per le seguenti session_ids: {list(store.keys())}"
    )
print("--- Fine Ispezione Memoria ---\n")
