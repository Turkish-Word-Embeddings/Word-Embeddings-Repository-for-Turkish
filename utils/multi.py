# combine wordvectors from word2vec and fasttext
from gensim.models import KeyedVectors
WORD2VEC = "../word2vec/word2vec_10epoch.wordvectors"
FASTTEXT = "../fasttext/fasttext-10ep.wordvectors"

word2vec_wv = KeyedVectors.load_word2vec_format(WORD2VEC, binary=True)
fasttext_wv = KeyedVectors.load_word2vec_format(FASTTEXT, binary=True)

model = KeyedVectors.load('combined1101107.wordvectors')

total = fasttext_wv.vectors.shape[0]
per = 10
step = total // per
print(step)

words = list(word2vec_wv.key_to_index.keys())
for idx, word in enumerate(words[1101108:]):
    print(idx + 1101108)
    v = word2vec_wv[word]
    w = fasttext_wv[word]
    vector = (v + w) / 2
    model.add_vectors(word, vector)
    if(idx % step == 0):
        model.save('combined{}.wordvectors'.format(idx)) 


model.save('combined.wordvectors')

        