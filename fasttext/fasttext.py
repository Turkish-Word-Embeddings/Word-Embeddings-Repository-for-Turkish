import argparse
import multiprocessing
import logging
import tempfile
import os, sys

# get the path of the directory containing the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from gensim.models.fasttext import FastText
from utils.utils import LineSentences
from utils.utils import callback

# paper: https://arxiv.org/pdf/1607.04606.pdf
# Assumption: Provided input is a txt file with one sentence per line.
# Example usage: python python fasttext/fasttext.py -i "corpus/bounwebcorpus.txt" "corpus/turkish-texts-tokenized.txt" -o "fasttext.model"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",  nargs='+', help="Input txt file. If file name includes space(s), enclose it in double quotes.", required=True )
    parser.add_argument("-o", "--output", help="Output file (trained model). If file name includes space(s), enclose it in double quotes.", default = "word2vec.model")
    parser.add_argument("-e", "--emb", help="dimensionality of word vectors, defaults to 300", default = 300, type = int)
    parser.add_argument("-w", "--ws", help="size of the context window, defaults to 5", default = 5, type = int)
    parser.add_argument("-ep", "--epoch", help="number of epochs, defaults to 5", default = 5, type = int)
    parser.add_argument("-sg", "--sg", help="use skip-gram model, defaults to 1 (Skip-gram)", default = 1, type = int)
    parser.add_argument("-hs", "--hs", help="use hierarchical softmax, defaults to 0 (negative sampling)", default = 0, type = int)
    parser.add_argument("-n", "--neg", help="number of negative samples, defaults to 5", default = 5, type = int)
    parser.add_argument("-l", "--lr", help="learning rate, defaults to 0.05", default = 0.05, type = float)
    parser.add_argument("-c", "--mincount", help="minimal number of word occurences, defaults to 10", default = 10, type = int)
    parser.add_argument("-mn", "--minn", help="minimal character ngram length, defaults to 3", default = 3, type = int)
    parser.add_argument("-mx", "--maxn", help="maximal character ngram length, defaults to 6", default = 6, type = int)
    parser.add_argument("-wng", "--wng", help="use word ngrams, defaults to 1", default = 1, type = int)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = FastText(window=args.ws, 
                sg = args.sg,
                hs = args.hs,
                negative=args.neg,
                alpha=args.lr,
                min_count=args.mincount,
                epochs=args.epoch,
                vector_size=args.emb,
                min_n=args.minn,
                max_n=args.maxn,
                word_ngrams=args.wng,
                workers=multiprocessing.cpu_count(),
                callbacks=[callback()])

    model.build_vocab(corpus_iterable=LineSentences(args.input))
    model.train(corpus_iterable=LineSentences(args.input), epochs = model.epochs, total_examples=model.corpus_count, compute_loss=True)
    loss = model.get_latest_training_loss()
    print("Loss: {}".format(loss)) 

    with tempfile.NamedTemporaryFile(prefix=args.output, delete=False) as tmp:   
        model.save(tmp.name, separately=[])
