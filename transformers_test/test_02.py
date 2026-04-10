import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import marimo as mo

    return mo, torch


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
    torch.manual_seed(123)
    embed = torch.nn.Embedding(10, 16)
    embed_sentence = embed(sentence).detach()
    x_2 = embed_sentence[1,:] 
    return embed_sentence, x_2


@app.cell
def _(embed_sentence, torch):
    d = embed_sentence.shape[1]
    #8 attention heads
    h = 8

    multihead_U_query = torch.rand(h,d,d)
    multihead_U_key   = torch.rand(h,d,d)
    multihead_U_value = torch.rand(h,d,d)
    return multihead_U_key, multihead_U_query, multihead_U_value


@app.cell
def _(multihead_U_key, multihead_U_query, multihead_U_value, x_2):
    multihead_U_query_2 = multihead_U_query.matmul(x_2)
    multihead_U_key_2 = multihead_U_key.matmul(x_2)
    multihead_U_value_2 = multihead_U_value.matmul(x_2)
    multihead_U_key_2[2]
    return


@app.cell
def _(embed_sentence):
    stacked_inputs = embed_sentence.T.repeat(8, 1, 1)
    stacked_inputs.shape
    return (stacked_inputs,)


@app.cell
def _(multihead_U_key, stacked_inputs, torch):
    multihead_keys = torch.bmm(multihead_U_key, stacked_inputs)
    multihead_keys = multihead_keys.permute(0, 2, 1)
    multihead_keys.shape
    return


@app.cell
def _(multihead_U_value, stacked_inputs, torch):
    multihead_values = torch.matmul(multihead_U_value, stacked_inputs)
    multihead_values = multihead_values.permute(0, 2, 1)
    return


@app.cell
def _(torch):
    multihead_z_2 = torch.rand(8, 16) #timeskip per semplicità
    return (multihead_z_2,)


@app.cell
def _(multihead_z_2, torch):
    linear = torch.nn.Linear(8*16, 16)
    context_vector_2 = linear(multihead_z_2.flatten())
    context_vector_2.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transformers
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
