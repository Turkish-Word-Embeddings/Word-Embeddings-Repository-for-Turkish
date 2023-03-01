import os
import gensim
fiil_ekleri = os.listdir("tasks/analogy/fiil çekim ekleri")
isim_ekleri = os.listdir("tasks/analogy/isim çekim ekleri")
yapim_ekleri = os.listdir("tasks/analogy/yapım ekleri")

experiments = [
    *[{"name": f, "filename": "fiil çekim ekleri/" + f, "topn": 10} for f in fiil_ekleri],
    *[{"name": f, "filename": "isim çekim ekleri/" + f, "topn": 10} for f in isim_ekleri],
    *[{"name": f, "filename": "yapım ekleri/" + f, "topn": 10} for f in yapim_ekleri],
    {"name": "yapım-çekim ekleri karışık.txt", "filename": "yapım-çekim ekleri karışık.txt", "topn": 10}
]

analogy_config = {
    "model_directory": "word2vec",
    "data_directory": "tasks/analogy",
    "model": {
        "type": "word2vec_keyedVectors", # or word2vec_keyedVectors
        "filename": "word2vec.model",
        "binary": True
    },
    "experiments": experiments
}

# load model

model_config = analogy_config["model"]

model_path = os.path.join(
    analogy_config["model_directory"],
    model_config["filename"])

if model_config["type"] == "word2vec_keyedVectors":
    model = gensim.models.KeyedVectors.load_word2vec_format(
        model_path,
        binary=model_config["binary"]
    )
elif model_config["type"] == "word2vec_model":
    model = gensim.models.Word2Vec.load(model_path).wv
elif model_config["type"] == "fasttext_model":
    model = loaded_model = gensim.models.fasttext.FastText.load(model_path).wv

class LineSentences(object):
    def __init__(self, filename):
        self.filename = os.path.join(analogy_config["data_directory"], filename)
    
    # memory-friendly iterator
    def __iter__(self):
        for line in open(self.filename, "r", encoding="utf-8"):
            yield line.strip().split()
            
def load_file(filename):
    f = open(
        os.path.join(
            analogy_config["data_directory"],
            filename),
        "r",
        encoding='utf-8')
    return [line.split() for line in f.readlines()]

def score_analogy(model, analogy, topn):
    """
    """
    results = model.most_similar_cosmul(
        positive=[analogy[1], analogy[2]],
        negative=[analogy[0]],
        topn=topn)
    
    return 1 / ([x[0] for x  in results].index(analogy[3]) + 1)

def run_experiment(model, experiment_config):
    """
    """
    analogies = load_file(experiment_config["filename"])
    
    mean_reciprocal_rank = 0
    repository_miss_ratio = 0
    topn_miss_ratio = 0

    for i, analogy in enumerate(analogies):
        
        try:
            mean_reciprocal_rank += score_analogy(
                model,
                analogy,
                experiment_config["topn"])
        except ValueError: # target word wasn't in the top n words
            topn_miss_ratio += 1
        except KeyError: # words in the analogy are not present in the repository
            repository_miss_ratio += 1
        except IndexError:
            print(f"IndexError: {analogy}")

    mean_reciprocal_rank = mean_reciprocal_rank / len(analogies)
    topn_miss_ratio = topn_miss_ratio / len(analogies)
    repository_miss_ratio = repository_miss_ratio / len(analogies)

    print("X"*50)
    print(f"EXPERIMENT `{experiment_config['name']}`")
    print(f"Mean reciprocal rank of experiment: {mean_reciprocal_rank}")
    print(f"topn miss ratio: {topn_miss_ratio}")
    print(f"repository miss ratio: {repository_miss_ratio}")
    print("X"*50)

for experiment in analogy_config["experiments"]:
    run_experiment(model, experiment)
