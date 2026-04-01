import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
import time

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell

class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks
    def __len__(self):
        return len(self.text_chunks)
    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()

def sample(model, starting_str,
           len_generated_text = 500,
           scale_factor = 1
          ):
    encoded_input = torch.tensor(
        [char2int[s] for s in starting_str]
    )
    encoded_input = torch.reshape(encoded_input, (1,-1))
    generated_str = starting_str

    model.eval() #set in mod eval -> alcuni moduli cambiano
    hidden, cell = model.init_hidden(1)
    for c in range(len(starting_str)-1): #passo la seq al modello, settando memoria e hidden state
        _,hidden,cell = model(encoded_input[:,c].view(1), hidden, cell)

    last_char = encoded_input[:,-1]
    for i in range(len_generated_text): #parto a generare
        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        generated_str += str(char_array[last_char])
    return generated_str

lang = 1 #0 = eng, 1 = 1ta

if lang == 0:
    with open('1268-0.txt', 'r', encoding='utf8') as fp:
        text = fp.read()

    start_indx = text.find('THE MYSTERIOUS ISLAND')
    end_indx   = text.find("End of the Project Gutenberg")
    text = text[start_indx:end_indx]
    char_set = set(text)

    print(start_indx)
    print(end_indx)
    print('Total Length (N° caratteri):', len(text))
    #print('Unique Characters:', len(char_set))
elif lang==1:
    with open('pg65391.txt', 'r', encoding='utf8') as fp:
        text = fp.read()

    start_indx = text.find('IL CONTE DI MONTE-CRISTO')
    end_indx   = text.find("END OF THE PROJECT GUTENBERG EBOOK")
    text = text[start_indx:end_indx]
    char_set = set(text)

    print(start_indx)
    print(end_indx)
    print('Total Length (N° caratteri): ', len(text))
    #print('Unique Characters:', len(char_set))

chars_sorted = sorted(char_set) #output dictionary
char2int = {ch:i for i,ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)
text_encoded = np.array(
    [char2int[ch] for ch in text],
    dtype=np.int32
)

seq_length = 40 #ci limitiamo a chunk di 41 caratteri
chunk_size = seq_length + 1
text_chunks = [text_encoded[i:i+chunk_size] for i in range(len(text_encoded)-chunk_size)]

seq_dataset = TextDataset(torch.tensor(text_chunks))

batch_size = 64
torch.manual_seed(42)
seq_dl = DataLoader(
    seq_dataset, 
    batch_size=batch_size,
    shuffle=True, 
    drop_last=True
)

vocab_size = len(char_array)
embed_dim = 256
rnn_hidden_size = 512
model = RNN(vocab_size, embed_dim, rnn_hidden_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

num_epochs = 3000
epoch_times = []
for epoch in range(num_epochs):
    start = time.time() #inizio musurazione tempo esecuzione
    #va settato model.train()?
    hidden, cell = model.init_hidden(batch_size)
    seq_batch, target_batch = next(iter(seq_dl))
    optimizer.zero_grad()
    loss = 0
    for c in range(seq_length):
        pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
        loss += loss_fn(pred, target_batch[:, c])
    loss.backward()
    optimizer.step()
    loss = loss.item()/seq_length
    end = time.time()
    epoch_times.append(end-start)
    if epoch % 50 == 0:
        print(f'Epoch {epoch} loss: {loss:.4f} | Avg exec time: {sum(epoch_times[-50:])/len(epoch_times[-50:]):.4f}')

print('--- Training concluso ---\nProcedo a salvare il modello')
torch.save(model.state_dict(), "output/models/ch2ch_model.pth")
print('Salvataggio completo\nAvvio test sample:\n')

print(sample(
    model, 
    starting_str='Vediamo un po cosa verrà fuori da questo modello che abbiamo addestrato per'
))