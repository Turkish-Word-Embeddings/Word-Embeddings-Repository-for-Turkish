import os
import gensim

analogy_config = {
    "model_directory": "tmp/model",
    "data_directory": "tmp/data",
    "model": {
        "type": "fasttext_model", # or word2vec_keyedVectors
        "filename": "english-fasttext.model",
        "binary": False
    },
    "experiments": [
        {
            "name": "general",
            "filename": "test.txt",
            "topn": 10
        }
    ]
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

    print(results)
    
    return 1 / ([x[0] for x  in results].index(analogy[3]) + 1)

def run_experiment(model, experiment_config):
    """
    """
    analogies = load_file(experiment_config["filename"])
    
    mean_reciprocal_error = 0
    repository_miss_ratio = 0
    topn_miss_ratio = 0

    for i, analogy in enumerate(analogies):
        
        try:
            mean_reciprocal_error += score_analogy(
                model,
                analogy,
                experiment_config["topn"])
        except ValueError: # target word wasn't in the top n words
            topn_miss_ratio += 1
        except KeyError: # words in the analogy are not present in the repository
            repository_miss_ratio += 1

    mean_reciprocal_error = mean_reciprocal_error / len(analogies)
    topn_miss_ratio = topn_miss_ratio / len(analogies)
    repository_miss_ratio = repository_miss_ratio / len(analogies)

    print("X"*50)
    print(f"EXPERIMENT `{experiment_config['name']}`")
    print(f"Mean reciprocal error of experiment: {mean_reciprocal_error}")
    print(f"topn miss ratio: {topn_miss_ratio}")
    print(f"repository miss ratio: {repository_miss_ratio}")
    print("X"*50)

for experiment in analogy_config["experiments"]:
    run_experiment(model, experiment)