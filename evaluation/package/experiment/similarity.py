import os
from .metadata import MetaData
from ..model.model import ModelWrapper
from ..log import MockLogger

def evaluate_similarity_folder(
        model: ModelWrapper,
        folder_path: str,
        file_type: str,
        extension: str,
        verbose: bool = True
):
    metadata_attrs = ["pearson", "pearson-p", "spearman", "spearman-p", "oov-ratio"]
    results = {}
    with MetaData(folder_path, file_type, metadata_attrs) as md:
        for file in md.files_in_folder:
            out = model.evaluate_word_pairs(
                os.path.join(folder_path, file),
                delimiter=None, encoding='utf8')
        
            pearson = out[0]        # Pearson correlation coefficient with 2-tailed p-value
            pearson_result = pearson[0] * 100
            spearman = out[1]       #  Spearman rank-order correlation coefficient between the similarities from the dataset and the similarities produced by the model itself, with 2-tailed p-value.
            oov_ratio = out[2]      #  Out-of-vocabulary ratio (The ratio of pairs with unknown words.)

            if verbose:
                print("-"*50)
                print(f"EXPERIMENT `{file}`")
                print(f"Pearson Result: {pearson_result} - p-value: {pearson[1]}")
                print(f"Spearman Result: {spearman[0] * 100} - p-value: {spearman[1]}")
                print(f"OOV Ratio: {oov_ratio}")
                print("-"*50)

            results[file] = {
                "pearson": pearson_result,
                "pearson-p": pearson[1],
                "spearman": spearman[0] * 100,
                "spearman-p": spearman[1],
                "oov-ratio": oov_ratio
            }

    file_name = os.path.normpath(folder_path).split(os.path.sep)[-1]
    with MetaData(".", "txt", metadata_attrs, file_name + extension, MockLogger()) as md:
        for file, result in results.items():
            md.update_metadata(file, result)

            