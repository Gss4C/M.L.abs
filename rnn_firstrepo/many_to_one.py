import pandas as pd
import numpy as np
from torch.utils.data.dataset import random_split
from torch import manual_seed
from datasets import load_dataset
import re
from collections import Counter, OrderedDict
from torch.utils.data import DataLoader
import torch
from torch import nn
import marimo as mo

class RNN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embed_dim, 
                 rnn_hidden_size, 
                 fc_hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(
            out,
            lengths.cpu().numpy(),
            enforce_sorted=False,
            batch_first=True
        )
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def train(dataloader):
    model.train() #set module in training mode
    total_acc , total_loss = 0,0

    for index, (text_batch, label_batch, lengths) in enumerate(dataloader):
        if index % 5 == 0:
            print(f'Sono al batch {index} del train')
        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:,0]
        loss = loss_fn(pred, label_batch.float())
        loss.backward()
        optimizer.step()
        total_acc += (
            (pred >= 0.5).float() == label_batch
        ).float().sum().item()
        total_loss += loss.item()*label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

def evaluate(dataloader):
    model.eval()
    total_acc, total_loss = 0,0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch.float())
            total_acc += (
                (pred>=0.5).float() == label_batch
            ).float().sum().item()
            total_loss += loss.item()*label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

class Vocab:
    def __init__(self, ordered_dict, specials=["<pad>", "<unk>"]):
        self.token2idx = {}
        # Prima i token speciali
        for i, token in enumerate(specials):
            self.token2idx[token] = i
        # Poi il vocabolario
        for i, token in enumerate(ordered_dict.keys(), start=len(specials)):
            self.token2idx[token] = i

        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.default_index = 1  # <unk>

    def __getitem__(self, token):
        return self.token2idx.get(token, self.default_index)

    def __len__(self):
        return len(self.token2idx)

    def lookup_token(self, idx):
        return self.idx2token.get(idx, "<unk>")

    def lookup_indices(self, tokens):
        return [self[t] for t in tokens]

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(
        '(?::|;|=)(?:-)?(?:\)|\(|D|P)',
        text.lower()
    )
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-','')
    tokenized = text.split()
    return tokenized

def collate_batch(batch):
    label_list, text_list, lengths = [],[],[]

    for item in batch:
        label_list.append(item['label']) #label già tensori, devo solo prenderle
        processed_text = torch.tensor(text_pipeline(item['text']), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))

    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)

    return padded_text_list, label_list, lengths

print('PRE-PROCESSING DEI DATI')

dataset = load_dataset("imdb")
dataset_dimension = 1000
print(f'Import del dataset, numero samples: {dataset_dimension}')
train_set = dataset["train"].select(range(dataset_dimension))  # max 25.000 esempi
test_dataset = dataset["test"].select(range(dataset_dimension))    # 25.000 esempi

# Ogni esempio è un dict: {"text": "...", "label": 0 or 1}
for example in train_set:
    text = example["text"]
    label = example["label"]  # 0 = neg, 1 = pos

manual_seed(1)
train_dataset, valid_dataset = random_split(
    list(train_set), 
    [800,200]
)

token_counts = Counter()
for rev in train_dataset:
    tokens = tokenizer(rev['text'])
    token_counts.update(tokens)

sorted_by_freq_tuples = sorted(
    token_counts.items(),
    key=lambda x: x[1], 
    reverse=True
)

ordered_dict = OrderedDict(sorted_by_freq_tuples)
vocab = Vocab(ordered_dict)

batch_size = 32
print(f'Comincio tokenizzazione del dataset e creazione dataloader con batch_size = {batch_size}')
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)

#TODO crea funzione create dataloader che contiene sto blocco di monnezza
train_dl = DataLoader(
    train_dataset, 
    batch_size = batch_size,
    shuffle    = True,
    collate_fn = collate_batch
)
valid_dl = DataLoader(
    valid_dataset, 
    batch_size = batch_size,
    shuffle    = True,
    collate_fn = collate_batch
)
test_dl = DataLoader(
    test_dataset, 
    batch_size = batch_size,
    shuffle    = True,
    collate_fn = collate_batch
)

vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64
torch.manual_seed(1)
model = RNN(vocab_size, embed_dim,rnn_hidden_size, fc_hidden_size)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 2
torch.manual_seed(1)

print(f'Inizio del training\nEpoche: {num_epochs}\nEmbed dim: {embed_dim}')

for epoch in range(num_epochs):
    acc_train, loss_train = train(train_dl)
    acc_valid, loss_valid = evaluate(valid_dl)
    print(f'Epoca: {epoch} accuracy: {acc_train:.4f}   -   val_accuracy: {acc_valid:.4f}')