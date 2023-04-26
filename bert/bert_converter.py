import torch

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

from transformers import AutoModel, AutoTokenizer

_model = "dbmdz/bert-base-turkish-128k-cased"
tokenizer = AutoTokenizer.from_pretrained(_model)
model = AutoModel.from_pretrained(_model)

def embed(word):
    marked_text = "[CLS] " + word + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():
        return model(tokens_tensor, segments_tensors)[1][0].numpy()

# word2vec
from gensim.models import KeyedVectors

kw = KeyedVectors(768)

vocab = list(tokenizer.vocab.keys())

batch_size = 100
number_of_batches = len(vocab) // batch_size + 1

for batch in range(number_of_batches):
    words_in_batch = vocab[:batch_size]
    words_in_batch = list(filter(lambda w: '#' not in w, words_in_batch))
    print(words_in_batch)
    vocab = vocab[batch_size:]

    embeddings = [embed(word) for word in words_in_batch]
    kw.add_vectors(words_in_batch, embeddings)

kw.save_word2vec_format("bert_word2vec_dump.txt")