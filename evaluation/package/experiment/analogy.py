
import os
from typing import List
from .metadata import MetaData
from ..model.model import ModelWrapper
from ..log import MockLogger

def score_analogy(model: ModelWrapper, analogy: List[str], topn: int):
    """
    """
    results = model.most_similar_cosmul(
        positive=[analogy[1], analogy[2]],
        negative=[analogy[0]],
        topn=topn)
    
    return 1 / ([x[0] for x  in results].index(analogy[3]) + 1)

def evaluate_analogy_folder(
        model: ModelWrapper,
        topn: int,
        folder_path: str,
        file_type: str,
        extension: str,
        verbose: bool = True
):
    
    results = {}
    with MetaData(folder_path, file_type) as md:
        for file in md.files_in_folder:
            analogies = md.load_file(file)
    
            mean_reciprocal_rank = 0
            repository_miss_ratio = 0
            topn_miss_ratio = 0

            for i, analogy in enumerate(analogies):
                
                try:
                    mean_reciprocal_rank += score_analogy(
                        model,
                        analogy,
                        topn)
                except ValueError: # target word wasn't in the top n words
                    topn_miss_ratio += 1
                except KeyError: # words in the analogy are not present in the repository
                    repository_miss_ratio += 1
                except IndexError:
                    print(f"IndexError: {analogy}")

            mean_reciprocal_rank = mean_reciprocal_rank / len(analogies)
            topn_miss_ratio = topn_miss_ratio / len(analogies)
            repository_miss_ratio = repository_miss_ratio / len(analogies)

            if verbose:
                print("-"*50)
                print(f"EXPERIMENT `{file}`")
                print(f"Mean reciprocal rank of experiment: {mean_reciprocal_rank}")
                print(f"topn miss ratio: {topn_miss_ratio}")
                print(f"repository miss ratio: {repository_miss_ratio}")
                print("-"*50)

            results[file] = mean_reciprocal_rank

    file_name = MetaData.get_folder_name(folder_path)
    with MetaData(".", "txt", file_name + extension, MockLogger()) as md:
        for file, result in results.items():
            md.update_metadata(file, result)

