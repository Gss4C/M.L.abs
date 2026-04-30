import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import groq
    import os
    from dotenv import load_dotenv
    return groq, load_dotenv, os


@app.cell
def _(load_dotenv, os):
    try:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("Key is None")
        print("API Key importata")
    except Exception as e:
        print(f'ERRORE nel recupero di api key: {e}')
    return (api_key,)


@app.cell
def _(api_key, groq):
    if api_key:
        try:
            client = groq.Groq(api_key=api_key)
            print('Creato client Groq')
        except groq.GroqError as e:
            print(f'Errore nella creazione del client Groq: {e}')
            client = None #per sicurezza
        except Exception as e:
            print(f'Errore: {e}')
            client = None
    else:
        print("Impossibile creare client, api key not found")
        client = None
    return (client,)


@app.cell
def _(client, groq):
    def chiama_groq(prompt_utente, max_tokens = 50, temperature =  None, top_p_val = None):
        if not client:
            print('Errore: client Groq non disponibile')
            return None
    
        #creo parametri per funzione, passarli con **kwargs
        params = {
          "model": "llama-3.3-70b-versatile",
          "messages": [
              {"role": "user", "content": prompt_utente}
          ],
          "max_tokens": max_tokens,
        }

        #gestione automatica top-p o temp
        if top_p_val is not None:
            params["top_p"] = top_p_val
            param_usato = f"top_p={top_p_val}"
        elif temperature is not None:
            params["temperature"] = temperature
            param_usato = f"temp={temperature}"
        else:
            param_usato = "default temp/top_p"
    
        print(f'\n--- Invio Prompt ---')
        print(f'Prompt utente: {prompt_utente}')
        print('--------------------')

        try:
            response = client.chat.completions.create(**params) #funziona uguale ad openai api
            testo_generato = response.choices[0].message.content.strip()

            print(f'-- Risposta del modello {response.model} --')
            print(testo_generato)
            print('--------------------')

            print(f'Token utilizzati: \nPrompt = {response.usage.prompt_tokens}\nCompletamento = {response.usage.completion_tokens}')
        except groq.APIError as e:
            print(f'Errore chiamata API: {e}')
        except Exception as e:
            print(f'Errore generico: {e}')

    return (chiama_groq,)


@app.cell
def _(chiama_groq):
    prompt_test = 'Spiega in breve perché un AI Engineer medio di oggi non sa niente di ML e di AI'
    chiama_groq(prompt_test, 100, temperature = 1.2)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
