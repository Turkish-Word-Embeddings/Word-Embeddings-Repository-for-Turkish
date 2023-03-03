import os
import gensim

experiments = [
    {"name": "similarity.txt", "filename": "similarity.txt"}
]

similarity_config = {
    "model_directory": "word2vec",
    "data_directory": "tasks",
    "model": {
        "type": "word2vec_keyedVectors", # or word2vec_keyedVectors
        "filename": "word2vec_10epoch.model",
        "binary": True
    },
    "experiments": experiments
}

# load model

model_config = similarity_config["model"]

model_path = os.path.join(
    similarity_config["model_directory"],
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
            similarity_config["data_directory"],
            filename),
        "r",
        encoding='utf-8')
    return [line.split(None) for line in f.readlines()]

def run_experiment(model, experiment_config):
    """
    """
    out = model.evaluate_word_pairs(os.path.join(similarity_config["data_directory"], experiment['filename']), delimiter=None, encoding='utf8')
    pearson = out[0]        # Pearson correlation coefficient with 2-tailed p-value
    spearman = out[1]       #  Spearman rank-order correlation coefficient between the similarities from the dataset and the similarities produced by the model itself, with 2-tailed p-value.
    oov_ratio = out[2]      #  Out-of-vocabulary ratio (The ratio of pairs with unknown words.)

    print("-"*50)
    print(f"EXPERIMENT `{experiment_config['name']}`")
    print(f"Pearson Result: {pearson[0] * 100} - p-value: {pearson[1]}")
    print(f"Spearman Result: {spearman[0] * 100} - p-value: {spearman[1]}")
    print(f"OOV Ratio: {oov_ratio}")
    print("-"*50)

for experiment in similarity_config["experiments"]:
    run_experiment(model, experiment)
