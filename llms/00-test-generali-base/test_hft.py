## TEST HUGGING-FACE TRANSFORMERS IN LOCALE ##
# ========================================== #

from transformers import pipeline
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    print("\nCaricamento pipeline Hugging Face...")
    
    generator = pipeline('text-generation', model='gpt2') #la task ora è text-generation, ma ce ne sono molte, vedere su HF
    prompt    = "This should be a prompt for the model we're testing"
    print(f'\nGenerazione testo a partire da: {prompt}')
    response  = generator(
        prompt, 
        max_length = 50,  # tokens
        num_return_sequences = 1
    )
    print('\nPrint risposta di GPT-2 locale')
    print(response[0]['generated_text'])
except ImportError:
    print('[ERRORE]')