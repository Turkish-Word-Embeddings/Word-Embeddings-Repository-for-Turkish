from .metadata import MetaData
from ..utils import get_yes_no_input
from .analogy import evaluate_analogy_folder
from .similarity import evaluate_similarity_folder

def add_metadata_file(folder: str, file_type: str, extension: str):
    
    with MetaData(
            folder,
            file_type,
            extension) as md:
        
        if md.metadata_exists:

            overwrite = get_yes_no_input("Do you want to overwrite the existing metadata file?")
            
            if overwrite == "yes":
                print("INFO: Existing metadata will be overwritten.")
            else:
                print("INFO: Exiting the program.")
                exit()
        
        for file in md.files_in_folder:
            score = ""
            while not isinstance(score, float):
                score = input(f" -> Reference score for `{file}`: ")
                try:
                    score = float(score)
                except:
                    print("WARNING: Enter a valid float.")
                    score = ""
            md.update_metadata(file, score)
        
        print(md._metadata)