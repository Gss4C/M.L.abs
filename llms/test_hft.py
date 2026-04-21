from transformers import pipeline
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    print("\nCaricamento pipeline Hugging Face...")
    
    generator = pipeline('text-generation', model='gpt2')
    prompt = "This should be a prompt"
    print(f'\nGenerazione testo a partire da: {prompt}')
    response = generator(
        prompt, 
        max_length = 50, 
        num_return_sequences = 1
    )
    print('\nPrint risposta di GPT-2 locale')
    print(response[0]['generated_text'])
except ImportError:
    print('[ERRORE]')