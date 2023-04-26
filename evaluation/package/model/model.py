import gensim

class ModelWrapper:

    def __getitem__(self, word):
        raise NotImplementedError

class Word2VecWrapper(ModelWrapper):

    def __init__(self, wv):
        self.wv = wv

    def __getitem__(self, word):
        return self.wv[word]
    
    def most_similar_cosmul(self, positive, negative, topn):
        return self.wv.most_similar_cosmul(
            positive=positive,
            negative=negative,
            topn=topn)
    
    def evaluate_word_pairs(self, file_path, delimiter, encoding):
        return self.wv.evaluate_word_pairs(
            file_path, delimiter=delimiter, encoding=encoding)

    @classmethod
    def from_keyed_vectors(cls, model_path, binary=True, no_header=False):
        wv = gensim.models.KeyedVectors.load_word2vec_format(
            model_path,
            binary=binary,
            no_header=no_header
        )
        return cls(wv)
    
    @classmethod
    def from_wordvectors(cls, model_path):
        wv = gensim.models.KeyedVectors.load(model_path)
        return cls(wv)

    @classmethod
    def from_model(cls, model_path):
        wv = gensim.models.Word2Vec.load(model_path).wv
        return cls(wv)
    
    @classmethod
    def from_fasttext(cls, model_path):
        wv = gensim.models.fasttext.FastText.load(model_path).wv
        return cls(wv)
    
    @classmethod
    def from_glove(cls, model_path):
        cls.from_keyed_vectors(model_path, False, True)
