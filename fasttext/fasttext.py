import fasttext
import argparse

# paper: https://arxiv.org/pdf/1607.04606.pdf
# Assumption: Provided input is a txt file with one sentence per line.
# Example usage: python fasttext/fasttext.py -i "corpus/Turkish-English Parallel Corpus.txt" -o "fasttext.model"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",  help="Input txt file. If file name includes space(s), enclose it in double quotes.", required=True )
    parser.add_argument("-o", "--output",  help="Output file (trained model). If file name includes space(s), enclose it in double quotes.", default = "word2vec.model")
    parser.add_argument("-m", "--model",  help="model type: cbow or skipgram, defaults to skipgram", default = "skipgram", type = str)
    parser.add_argument("-d", "--dim",  help="dimensionality of word vectors, defaults to 128", default = 128, type = int)
    parser.add_argument("-w", "--ws",  help="size of the context window, defaults to 5", default = 5, type = int)
    parser.add_argument("-e", "--epoch",  help="number of epochs, defaults to 5", default = 5, type = int)
    parser.add_argument("-n", "--neg",  help="number of negative samples, defaults to 5", default = 5, type = int)
    parser.add_argument("-l", "--lr",  help="learning rate, defaults to 0.05", default = 0.05, type = float)
    parser.add_argument("-c", "--mincount",  help="minimal number of word occurences, defaults to 5", default = 5, type = int)
    parser.add_argument("-mn", "--minn",  help="minimal character ngram length, defaults to 3", default = 3, type = int)
    parser.add_argument("-mx", "--maxn",  help="maximal character ngram length, defaults to 6", default = 6, type = int)
    parser.add_argument("-wng", "--wng",  help="use word ngrams, defaults to 1", default = 1, type = int)
    parser.add_argument("-ls", "--loss",  help="loss function, defaults to ns", default = "ns", type = str)
    args = parser.parse_args()

    # train the model
    model = fasttext.train_unsupervised(input = args.input,  
                                    model= args.model,
                                    dim = args.dim,
                                    ws = args.ws,
                                    epoch = args.epoch,
                                    neg = args.neg,
                                    lr = args.lr,
                                    minCount = args.mincount,
                                    minn = args.minn,
                                    maxn = args.maxn,
                                    wordNgrams = args.wng,
                                    loss = args.loss)
    # save the model
    model.save_model(args.output)
