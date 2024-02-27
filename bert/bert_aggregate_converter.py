# run like:
# python bert_aggregate_convert.py <input_file> <vocab_file>

from sys import argv
assert len(argv) == 3, "should run like 'python bert_aggregate_convert.py <input_file> <vocab_file>'"

import re
from typing import Dict, List
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

_model = "dbmdz/bert-base-turkish-128k-cased"
tokenizer = AutoTokenizer.from_pretrained(_model)
model = AutoModel.from_pretrained(_model)

def get_embedding(sentence: str):
    # marked_text = "[CLS] " + word + " [SEP]"
    # CLS aradaki cümlenin embedding'i olacak
    marked_text = "[CLS] " + sentence + " [SEP]"

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
        results = model(tokens_tensor, segments_tensors)
        return results[0][0, 0].numpy()

class Sentence:
    def __init__(self, sentence: str):
        self.sentence = sentence
        self._embedding = None
        self._embedding_cached = False
    
    @property
    def embedding(self):
        if self._embedding_cached:
            return self._embedding
        
        embedding = get_embedding(self.sentence)
        self._embedding_cached = True
        self._embedding = embedding
        return embedding

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        vocab = map(lambda line: line.strip().split()[0], lines)
    return set(list(vocab))

vocab_path = argv[2]
vocab = load_vocab(vocab_path)

# open the file
filename = argv[1]
with open(filename, "r") as file:
    # create a dictionary to store the word index
    index: Dict[str, List[Sentence]] = {}
    # read the file line by line
    for line in tqdm(file, desc="Reading File"):
        # split the line into sentences
        sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', line.strip())
        # iterate over each sentence
        for sentence in sentences:
            sentence = Sentence(sentence)
            # split the sentence into words
            words = sentence.sentence.split()
            # iterate over each word
            for word in words:
                # if word is not in vocabulary, skip
                if word not in vocab:
                    continue

                # if the word is not in the index yet, create a new list for it
                if word not in index:
                    index[word] = []
                # add the sentence to the list for this word
                index[word].append(sentence)

out_file = open("vector_dump.txt", "w", encoding="utf-8")
for word, sentences in tqdm(index.items(), desc="Writing vectors"):
    acc = None
    for i, sentence in enumerate(sentences):

        if i==0:
            acc = sentence.embedding
        else:
            acc += sentence.embedding

    acc /= len(sentences)

    out_file.write(f"{word} {' '.join([str(v) for v in acc])}\n")
