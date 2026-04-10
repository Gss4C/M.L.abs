import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import transformers as tf

    return (tf,)


@app.cell
def _(tf):
    generator = tf.pipeline('text-generation', model = 'gpt2')
    return (generator,)


@app.cell
def _(generator, tf):
    tf.set_seed(123)
    generator('Questo modello maledetto funziona bene solamente in inglese',
              max_length=25,
              num_return_sequences = 3)
    return


@app.cell
def _():
    from transformers import GPT2Tokenizer
    from transformers import GPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text = "Let us encode this sentence"
    encoded_input = tokenizer(text, return_tensors = 'pt')
    print(encoded_input)

    model = GPT2Model.from_pretrained('gpt2')
    output = model(**encoded_input)
    print(output['last_hidden_state'].shape)
    #print(output['last_hidden_state'])
    return (output,)


@app.cell
def _(output):
    print(output['last_hidden_state'])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
