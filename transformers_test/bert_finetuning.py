import gzip
import shutil
import time
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset
import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
import marimo as mo

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def split_dataloader(train_dataset, valid_dataset, test_dataset, batch_size):
    train_dl = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = batch_size,
        shuffle    = True
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size = batch_size,
        shuffle    = True
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = batch_size,
        shuffle    = True
    )
    return train_dl, valid_dl, test_dl

def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0,0
        for batch_idx, batch in enumerate(data_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs          = model(input_ids, attention_mask=attention_mask)
            logits           = outputs['logits']
            predicted_labels = torch.argmax(logits, 1)
            num_examples    += labels.size(0)
            correct_pred    += (predicted_labels == labels).sum()
        return correct_pred.float()/num_examples * 100

torch.backends.cudnn.deterministic = True
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 3

url = ("https://github.com/rasbt/"
 "machine-learning-book/raw/"
 "main/ch08/movie_data.csv.gz")
filename = url.split("/")[-1]

with open(filename, "wb") as f:
    r = requests.get(url)
    f.write(r.content)

with gzip.open('movie_data.csv.gz', 'rb') as f_in:
    with open('movie_data.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

df = pd.read_csv('movie_data.csv')

#dataset e tokenizazione
train_texts  = df.iloc[:35000]['review'].values
train_labels = df.iloc[:35000]['sentiment'].values

valid_texts  = df.iloc[35000:40000]['review'].values
valid_labels = df.iloc[35000:40000]['sentiment'].values

test_texts  = df.iloc[40000:]['review'].values
test_labels = df.iloc[40000:]['sentiment'].values

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
valid_encodings = tokenizer(list(valid_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

train_dataset = IMDbDataset(train_encodings, train_labels)
valid_dataset = IMDbDataset(valid_encodings, valid_labels)
test_dataset  = IMDbDataset(test_encodings, test_labels)

train_loader, valid_loader, test_loader= split_dataloader(train_dataset, valid_dataset, test_dataset, batch_size = 16)

model = DistilBertForSequenceClassification.from_pretrained( 'distilbert-base-uncased')
model.to(DEVICE)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=5e-5)

# inizio training
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        # preparazione dati
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels         = batch['labels'].to(DEVICE)

        #forward pass
        outputs = model(
            input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        loss, logits = outputs['loss'], outputs['logits'] #output std di distilbert/seq clf
        #backward
        optim.zero_grad()
        loss.backward()
        optim.step()

        #log
        if not batch_idx % 250:
            print(
                f'Epoca: {epoch+1:04d}/{NUM_EPOCHS:04d}'
                f' | Batch {batch_idx:04d}/{len(train_loader):04d} | '
                f'Loss: {loss:.4f}'
            )
    model.eval()
    
    with torch.set_grad_enabled(False):
        print(
            f'Training Accuracy: {compute_accuracy(model, train_loader, DEVICE):.2f}%'
            f'\nValid Accuracy:    {compute_accuracy(model, valid_loader, DEVICE):2f}%'
            f'\nTempo Trascorso:   {(time.time() - start_time)/60:.2f} min'
        )
print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')
print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')