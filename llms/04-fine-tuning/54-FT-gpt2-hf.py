from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from transformers import pipeline

def prepare_data(examples):
    # Tokenizziamo il testo
    inputs = tokenizer(
        examples["text"], 
        truncation = True,         # Taglia testi len > max_length
        padding    = "max_length", # Aggiunge padding se non raggiungo max_length
        max_length = 128           # Lunghezza massima token
    )
    # Per il training causale, labels = input_ids
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# WikiText-2 è un dataset di testi da Wikipedia
print("\n Download dataset WikiText-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# solo 500 samples
small_train = dataset["train"].select(range(500))
small_eval  = dataset["validation"].select(range(100))
print(f"Dataset pronto - Training: {len(small_train)} esempi")

# Carichiamo tokenizer e modello
print("\nCaricamento GPT-2...")
tokenizer           = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Token per padding
model               = GPT2LMHeadModel.from_pretrained("gpt2")
print("Modello e tokenizer caricati")

print("\n Preparazione dati...")
train_data = small_train.map(prepare_data, batched=True)
eval_data  = small_eval.map(prepare_data,  batched=True)
print("Dati preparati.")

print("\nConfigurazione training...")
training_args = TrainingArguments(
    output_dir                  = "./risultati", # Dove salvare il modello
    num_train_epochs            = 1,                
    per_device_train_batch_size = 4,     
    per_device_eval_batch_size  = 4,      
    warmup_steps                = 100,     # lr aumenta ogni 100 steps
    logging_steps               = 50,      # Log ogni 50 step
    save_steps                  = 500,     # Salva ogni 500 step
    eval_steps                  = 100,     # Valuta ogni 100 step
)
trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_data,
    eval_dataset  = eval_data,
    processing_class = tokenizer,
)

print("\n=== Inizio training ===")
trainer.train()
print("\n=== Training completato ===")

# Salviamo il modello
print("\nSalvataggio modello...")
model.save_pretrained("./gpt2-FTD")
tokenizer.save_pretrained("./gpt2-FTD")
print("Modello salvato in ./gpt2-FTD")

# ===========================
# === Testiamo il modello ===
# ===========================
            
print("\n\n******************************\n******************************\nTest del modello fine-tuned...")
# Creiamo un generatore di testo
generator = pipeline(
    "text-generation", 
    model     = "./gpt2-FTD",
    tokenizer = tokenizer)

# Proviamo alcuni prompt
prompts = [
    "The meaning of life is",
    "Artificial intelligence will",
    "In the future, humans"
]

print("\n Esempi di generazione:")
print("="*50)
for prompt in prompts:
    # Generiamo testo
    result = generator(
        prompt, 
        max_length  = 50,      
        temperature = 0.8,    
        do_sample   = True      # Campionamento probabilistico
    )
    print(f"\nPrompt: {prompt}")
    print(f"Output: {result[0]['generated_text']}")
    print("-"*50)

print("\nCompleto! Modello salvato e testato in ./gpt2-FTD")