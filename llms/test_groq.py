import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv() #carica variabili in .env

#esempio di come inizializzare
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print('[ERRORE]: API Key not found')
else:
    try:
        os.getenv("GROQ_API_KEY")
        client = Groq()
        print('Invio API-request')
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Sei un robot che risponde sintetico ed essenziale"},#indicazioni al modello
                {"role": "user","content": "dimmi perché llama 3.3 70b è meglio di gpt5"}
            ],
            max_tokens = 500,
            temperature = 0.7 #randomness del modello. 0 è deterministico
        )

        print('Risposta del modello')
        print(response.choices[0].message.content.strip())

    except:
        print('[ERRORE]: ma non so perché')