import os
from google.colab import userdata  # da cambiare per usarlo local
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import llm_result

hft_api_key = userdata.get("HFT_KEY")  # da cambiare per usarlo local
if hft_api_key:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hft_api_key
    print("Preso token api di HFT")
else:
    print("Errore nel recupero di api token di HFT")


repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError("Missing HFT_KEY environment variable")
    LLM_HF = None
else:
    llm_base = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.7,
        max_new_tokens=128,
        do_sample=True,
    )

LLM_HF = ChatHuggingFace(llm=llm_base)

prompt_template_string = """
Suggerisci un nome creativo per un'azienda che produce: {product_description}

Nome:
"""
prompt = ChatPromptTemplate.from_template(prompt_template_string)
# prompt = PromptTemplate.from_template(prompt_template_string)

print("Prompt template creato.")

# la chain viene creata attaccando componenti tramite operatore Pipe

chain = prompt | LLM_HF | StrOutputParser()
print("Chain creata con successo.")

input_data = {"product_description": "Cioccolato artigianale biologico"}

print("\n--- Esecuzione Chain HF ---")

result = chain.invoke(input_data)
print(f"Output da {repo_id}: {result.strip()}")


# posso farlo anche con altri componenti, cambia molto poco. La chain è model-agnostic

from langchain_groq import ChatGroq

groq_apikey = userdata.get("GROQ_API_KEY")

if groq_apikey:
    os.environ["GROQ_API_KEY"] = groq_apikey
    print("Token API ritrovato")
else:
    print("Nessun token API trovato")

print(os.environ["GROQ_API_KEY"])

llm_groq = None

llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)
print(f"Inizializzato modello {llm_groq.model_name}")


if llm_groq and prompt:
    chain_groq = prompt | llm_groq | StrOutputParser()
    print("Chain Groq creata con successo.")

    print("--- Eseguo la chain ---")
    result = chain_groq.invoke(input_data)
    print(f"Output da {repo_id}: {result.strip()}")
