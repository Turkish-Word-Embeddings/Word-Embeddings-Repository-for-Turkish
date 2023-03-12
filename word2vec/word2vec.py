import argparse
import multiprocessing
import logging
import os, sys

# get the path of the directory containing the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from utils.utils import LineSentences
from utils.utils import callback

# Assumption: Provided input is a txt file with one sentence per line.
# Example usage: python word2vec/word2vec.py -i "corpus/bounwebcorpus.txt" -o "word2vec.model"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", nargs='+', help="Input txt file. If file name includes space(s), enclose it in double quotes.", required=True)
    parser.add_argument("-o", "--output",  help="Output file (trained model). If file name includes space(s), enclose it in double quotes.", default = "word2vec.model")
    parser.add_argument("-m", "--min_count",  help="Minimum frequency for a word. All words with total frequency lower than this will be ignored. Defaults to 10 if not provided.", default = 10, type = int)
    parser.add_argument("-e", "--emb",  help="dimensionality of word vectors, defaults to 300", default = 300, type = int)
    parser.add_argument("-w", "--window",  help="window size, defaults to 5", default = 5, type = int)
    parser.add_argument("-ep", "--epoch",  help="number of epochs, defaults to 5", default = 5, type = int)
    parser.add_argument("-sg", "--sg",  help="use skip-gram model, defaults to 1 (Skip-gram)", default = 1, type = int)
    parser.add_argument("-hs", "--hs",  help="use hierarchical softmax, defaults to 0 (negative sampling)", default = 0, type = int)
    parser.add_argument("-n", "--negative",  help="number of negative samples, defaults to 5", default = 5, type = int)
    args = parser.parse_args()
    
    input = args.input
    output = args.output
    emb = args.emb
    window = args.window
    epoch = args.epoch
    sg = args.sg
    hs = args.hs
    negative = args.negative
    min_count = args.min_count

    """
    Word2vec accepts several parameters that affect both training speed and quality.
    * min_count: This will ignore all words with total frequency lower than this.
      Words that appear only once or twice in a billion-word corpus are probably uninteresting typos and garbage.
    * workers: Parallelization. The workers parameter has only effect if you have Cython installed.
    * sentences: The sentences iterable can be simply a list of lists of tokens, but for larger corpora, consider an iterable that streams the sentences directly from disk/network. See `BrownCorpus`, `Text8Corpus` or `LineSentence` in word2vec module for such examples. See also the tutorial on data streaming in Python. If you don’t supply sentences, the model is left uninitialized – use if you plan to initialize it in some other way.
    * vector_size: Dimensionality of the word vectors.
    * window: Maximum distance between the current and predicted word within a sentence.
    * sg: Training algorithm: 1 for skip-gram; otherwise CBOW.
    * hs: If 1, hierarchical softmax will be used for model training. If 0, and negative is non-zero, negative sampling will be used.
    * negative: If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(sentences=LineSentences(input), 
                vector_size=emb, 
                window=window, 
                min_count=min_count, 
                epochs = epoch, 
                sg = sg,
                hs = hs,
                negative = negative,
                compute_loss=True,
                workers=multiprocessing.cpu_count(),
                callbacks=[callback()])
    # The full model can be stored/loaded via its save() and load() methods.
    model.wv.save_word2vec_format(output, binary=True)

    word_vectors = KeyedVectors.load_word2vec_format(output, binary=True)
    word_vectors.most_similar_cosmul(positive=['kadın', 'kral'], negative=['adam'])