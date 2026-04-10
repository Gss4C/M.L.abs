import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import marimo as mo

    return F, mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic Self-Attention mechanism

    Test dove seguo il libro per capire come funzionano i transformrs
    """)
    return


@app.cell
def _(torch):
    sentence = torch.tensor(
     [0, # can
     7, # you
     1, # help
     2, # me
     5, # to
     6, # translate
     4, # this
     3] # sentence
    )
    sentence
    return (sentence,)


@app.cell
def _(sentence, torch):
    torch.manual_seed(123)
    embed = torch.nn.Embedding(10, 16)
    embed_sentence = embed(sentence).detach()
    embed_sentence.shape
    return (embed_sentence,)


@app.cell
def _(embed_sentence):
    embed_sentence
    return


@app.cell
def _(embed_sentence, torch):
    omega = torch.empty(8,8) #similarity based weights
    for i, xi in enumerate(embed_sentence):
        for j, xj in enumerate(embed_sentence):
            omega[i,j] = torch.dot(xi, xj)

    #cod inefficiente
    #tipicamente si fa
    omega_mat = embed_sentence.matmul(embed_sentence.T)
    torch.allclose(omega, omega_mat)
    return (omega,)


@app.cell
def _(F, omega):
    attention_weights = F.softmax(omega, dim=1)
    attention_weights.shape
    return (attention_weights,)


@app.cell
def _(attention_weights):
    attention_weights.sum(dim=1) #verifica normalizzazione
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Qui ho fatto finora il calcolo del vettore omega che contiene tutti i prodotti di tutti gli elementi di x con tutti gli elementi di x. Ovviamente per avere il contesto adattto ad un singolo valore nella seuqenza devo prendere una sola riga della matrice omega. Questo mi darà il context vector per quel singolo elemento, la sequenza di elementi di context l'ottengo usando tutto.

    Calcoliamo il singolo vettore di contesto $\mathbf{z}^{(i)} = \sum\limits_{j=1}^T \alpha_{ij}\mathbf{x}^j$
    """)
    return


@app.cell
def _(attention_weights, embed_sentence, torch):
    #prendo ad esempio x_2
    x_2 = embed_sentence[1,:] 
    context_vec_2 = torch.zeros(x_2.shape)
    for _j in range(8):
        x_j = embed_sentence[_j , :]
        context_vec_2 += attention_weights[1, _j] * x_j

    context_vec_2
    return (context_vec_2,)


@app.cell
def _(attention_weights, embed_sentence, torch):
    #genneralizzo senza for loops ma con matmul
    context_vectors = torch.matmul(attention_weights, embed_sentence)
    return (context_vectors,)


@app.cell
def _(context_vec_2, context_vectors, torch):
    torch.allclose(context_vec_2, context_vectors[1])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
