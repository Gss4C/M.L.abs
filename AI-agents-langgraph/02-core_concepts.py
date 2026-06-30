from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model #metodo moderno

load_dotenv()

def demo_basic_chain(question: str, model):
    #esempio di costruzone di una chain con LCEL

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )
    llm_endpoint = HuggingFaceEndpoint(
        repo_id            = "meta-llama/Llama-3.1-8B-Instruct",
        task               = "text-generation",
        max_new_tokens     = 256,
        do_sample          = False,
        repetition_penalty = 1.03,
        provider           = "auto",  
    )
    model = ChatHuggingFace(llm=llm_endpoint, temperature=0.5)

    parser = StrOutputParser()
    
    chain = prompt | model | parser

    # Execute the chain with an input
    result = chain.invoke({"question": question})
    print(f"Response: {result}")

    return chain

def demo_batch_exectution(model):
    '''
    Esempio di come si può preparare un'esecuzione per input multipli
    '''
    prompt = ChatPromptTemplate.from_template('Traduci in francese il seguente testo: {testo}')
    parser = StrOutputParser()
    chain = prompt | model | parser

    #batch - run multiple inputs
    inputs = [
        {"testo": "Qusto è il primo test di traduzione in francese"}, 
        {"testo": "Spero di star capendo bene questi concetti"}
    ]
    results = chain.batch(inputs)
    for text in zip(inputs, results):
        print(f"Input: {text[0]['testo']} => Output: {text[1]}")

def demo_streaming(model):
    """Demonstrate streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")

    parser = StrOutputParser()

    chain = prompt | model | parser

    # Streaming - run with streaming enabled
    print("Streaming output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming

def demo_schema_inspection(model):
    """Demonstrate input/output schema inspection.
    Questo mi permette di sapere come mi apsetto le cose
    in input ed output dalla chain
    """
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Inspect input and output schemas
    input_schema  = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")

def exercise(model, name, audience):
    """
    EXERCISE: Create a chain that:
    1. Takes a product name and target audience
    2. Generates a marketing tagline
    3. Returns just the tagline as a string

    Test with: product="AI Course", audience="developers"
    """
    prompt = ChatPromptTemplate.from_template(
        'You are a marketing expert, generate three tagline for the product named {name} for the audience {audience}'
    )
    parser = StrOutputParser()

    chain = prompt | model | parser
    result = chain.invoke({"name": name, "audience": audience})
    print(f"Risposta chatbot: {result}")
    return chain

def new_way():
    # the univeral way to initialize a model
    # questo metodo moderno è più usato oggi
    model = init_chat_model("gpt-4o-mini", temperature=0.7, max_tokens=1500)
    #devo fare test e vedere se funziona anche con hf

if __name__ == "__main__":
        
    llm_endpoint = HuggingFaceEndpoint(
        repo_id            = "meta-llama/Llama-3.1-8B-Instruct",
        task               = "text-generation",
        max_new_tokens     = 256,
        do_sample          = False,
        repetition_penalty = 1.03,
        provider           = "auto",  
    )
    model = ChatHuggingFace(llm=llm_endpoint, temperature=0.5)

    #demo_basic_chain(question = "a cosa serve langchain?", model = model)
    #demo_batch_exectution(model)
    #demo_streaming(model)
    #demo_schema_inspection(model)
    exercise(model = model, name = "AI Course", audience = "Developers")