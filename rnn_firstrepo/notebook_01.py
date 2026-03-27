import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
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

    return (
        Counter,
        DataLoader,
        OrderedDict,
        load_dataset,
        manual_seed,
        mo,
        nn,
        random_split,
        re,
        torch,
    )


@app.cell
def _(load_dataset):
    dataset = load_dataset("imdb")
    dataset_dimension = 1000
    train_set = dataset["train"].select(range(dataset_dimension))  # max 25.000 esempi
    test_dataset = dataset["test"].select(range(dataset_dimension))    # 25.000 esempi

    # Ogni esempio è un dict: {"text": "...", "label": 0 or 1}
    for example in train_set:
        text = example["text"]
        label = example["label"]  # 0 = neg, 1 = pos
        #print("testo: ", text)
        #print("label: ", label)
    return test_dataset, train_set


@app.cell
def _(train_set):
    for i, sample in enumerate(train_set):
        if i<10:
            print("Review: ", sample['text'])
            print('Label: ', sample['label'])
        else: break
    return


@app.cell
def _(train_dataset, train_set):
    print(train_set['label'].count(0))
    train_dataset
    return


@app.cell
def _(manual_seed, random_split, train_set):
    manual_seed(1)
    train_dataset, valid_dataset = random_split(
        list(train_set), [800,200]
    )
    return train_dataset, valid_dataset


@app.cell
def _(re):
    # Def funzioni di tokenizzazione, voglio contare le occorrenze delle singole parole
    def tokenizer(text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall(
            '(?::|;|=)(?:-)?(?:\)|\(|D|P)',
            text.lower()
        )
        text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-','')
        tokenized = text.split()
        return tokenized

    return (tokenizer,)


@app.cell
def _(train_dataset):
    train_dataset[0]
    return


@app.cell
def _(Counter, tokenizer, train_dataset):
    token_counts = Counter()
    for rev in train_dataset:
        tokens = tokenizer(rev['text'])
        token_counts.update(tokens)
    print('Vocab-size: ', len(token_counts))
    print(token_counts.most_common(10))
    return (token_counts,)


@app.cell
def _():
    #provo a costruirmi a mano una classe da torchtext
    #questa qui fa da sola direttamente il vecchio seguente comando:
    '''
    vocab = vocab(ordered_dict)
    vocab.insert_token("<pad>", 0)
    vocab.insert_token("<unk>", 1)
    vocab.set_default_index(1)
    '''
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


    return (Vocab,)


@app.cell
def _(OrderedDict, Vocab, token_counts):
    sorted_by_freq_tuples = sorted(
        token_counts.items(),
        key=lambda x: x[1], 
        reverse=True
    )

    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocab = Vocab(ordered_dict)
    return (vocab,)


@app.cell
def _(vocab):
    print([vocab[token] for token in ['this', 'is', 'an', 'example']])
    return


@app.cell
def _(nn, tokenizer, torch, vocab):
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

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

    return (collate_batch,)


@app.cell
def _(DataLoader, collate_batch, train_dataset):
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)
    #dataloader = DataLoader(train_data, batch_size=4, shuffle=False)
    return (dataloader,)


@app.cell
def _(dataloader):
    next(iter(dataloader)) #ok questo ha senso
    len(dataloader)
    return


@app.cell
def _(DataLoader, collate_batch, test_dataset, train_dataset, valid_dataset):
    # creo i dataloader che mi servono
    batch_size = 32

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
    return train_dl, valid_dl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Costruzione delle RNN
    Farò oiù varianti
    - RNN semplice
    - LSTM
    - GRU

    NB: necessario un data pre-processing opzionale ma molto raccomandato per riuscire ad abbassare la dimensionalità del dataset
    """)
    return


@app.cell
def _(nn, torch, vocab):
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

    vocab_size = len(vocab)
    embed_dim = 20
    rnn_hidden_size = 64
    fc_hidden_size = 64
    torch.manual_seed(1)
    model = RNN(vocab_size, embed_dim,rnn_hidden_size, fc_hidden_size)
    model
    return (model,)


@app.cell
def _(loss_fn, model, optimizer, torch):
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

    return evaluate, train


@app.cell
def _(train_dl):
    for indice, (text_batch, label_batch, lengths) in enumerate(train_dl):
        print(indice)
    return


@app.cell
def _(model, nn, torch):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 2
    torch.manual_seed(1)
    return loss_fn, num_epochs, optimizer


@app.cell
def _(evaluate, num_epochs, train, train_dl, valid_dl):
    # prmo train
    for epoch in range(num_epochs):
        acc_train, loss_train = train(train_dl)
        acc_valid, loss_valid = evaluate(valid_dl)
        print(f'Epoca: {epoch} accuracy: {acc_train:.4f}   -   val_accuracy: {acc_valid:.4f}')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
