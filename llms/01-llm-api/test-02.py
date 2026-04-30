import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from dotenv import load_dotenv
    import os
    import groq
    import json
    return groq, json, load_dotenv, os


@app.cell
def _(load_dotenv, os):
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    return (api_key,)


@app.cell
def _(api_key, groq):
    #controllo
    if not api_key:
        print('[ERRORE]: API Key not found')
    else:
        try:
            client = groq.Groq(api_key=api_key)
        except Exception as e:
            print('Errore caricamento api key')
    return (client,)


@app.cell
def _():
    #voglio estrarre info da un testo scritto in naturale
    testo_input = """
    Gentile Sig. Mario Rossi,
    La contattiamo riguardo al suo ordine #12345.
    Può contattarci a support@example.com per assistenza.
    Cordiali saluti,
    Servizio Clienti.
    """
    #dato il testo sopra, preparo prompt per fargli fare le cose bene
    #faccio un one-shot learning
    prompt_per_json = f"""
    dato il seguente testo:
    ---
    {testo_input}
    ---
    estrai nome del cliente ed inidirizzo email.
    Rispondi **SOLO** con un oggetto json valido che contenga le chiavi "nome_cliente" e "email_contatto". 
    Non aggiungere nient'altro prima e dopo il JSON.
    Esempio formato atteso: {{"nome_cliente": "Nome e cognome", "email_contatto": "email@example.com"}}
    """

    print(prompt_per_json)
    return (prompt_per_json,)


@app.cell
def _(client, json, prompt_per_json):
    try:
        response = client.chat.completions.create( #si può usare funx di test-01
            model       = "llama-3.3-70b-versatile",
            messages    = [{"role": "user", "content": prompt_per_json}],
            temperature = 0.1
        )
        raw_text_response = response.choices[0].message.content.strip()
        print("\n--- Risposta grezza del modello ---")
        print(raw_text_response)

        print("\n Parsing JSON")
        dati_estratti = None
        try:
            dati_estratti = json.loads(raw_text_response)
            print("JSON parsato con successo\nEstrazione info da json")
            nome  = dati_estratti.get("nome_cliente", "Non Trovato")
            email = dati_estratti.get("email_contatto", "Non Trovato") 
            print(f"Nome estratto: {nome}")
            print(f"Email estratta: {email}")
        except json.JSONDecodeError:
            print("ERRORE: La risposta del modello non è un json valido!\nImpossibile fare parsing")
        except Exception as e:
            print(f"ERRORE: Errrore generico durante il parsing: {e}")
    except Exception as e:
        print(f"Verificato un errore durante la chiamata al modello: {e}")
    
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
