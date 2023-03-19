import os
import json
from .model import *
from ..utils import get_yes_no_input

class Model_MetaData:

    def __init__(self, model_path: str, metadata_extension: str = ".nlp_metadata"):
        self.model_path = model_path
        self.metadata_extension = metadata_extension

    @property
    def metadata_path(self):
        return self.model_path + self.metadata_extension

    def load_metadata(self):
        if not os.path.isfile(self.metadata_path):
            print(
                f"WARNING: No metadata about the model `{self.model_path}`"
                "was found. Please configure how the model is to be loaded below.")
            metadata = self.configure_metadata()
            print(f"INFO: Saving the new metadata to `{self.metadata_path}`")
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f)
                return metadata
        else:
            with open(self.metadata_path, "r") as f:
                return json.load(f)

    def load_model(self) -> ModelWrapper:
        
        if not os.path.isfile(self.model_path):
            raise ValueError("No model found in the path given"
                             f"{self.model_path}")
        
        metadata = self.load_metadata()
        kls = globals()[metadata["class"]]
        method = getattr(kls, metadata["method"])
        return method(**metadata["parameters"])
                
    def configure_metadata(self):
        while True:
            parameters = {"model_path": self.model_path}
            kls = input(" -> Class of the model (see classes in `model.py`): ")
            method = input(" -> class method which loads the model: ")
            
            while True:
                not_done = get_yes_no_input(f"Will this method require parameters in addition to `{parameters.keys()}`?")
                
                if not_done == "yes":
                    param = input("    -> Parameter name: ")
                    param_type = input("    -> Parameter type (float/bool/str): ")
                    if param_type == "bool":
                        print("       Enter 1 for True, 0 for False")
                    value = input("    -> Parameter value: ")

                    if param_type == "float":
                        value = float(value)
                    elif param_type == "bool":
                        value = bool(int(value))
                    parameters[param] = value
                else:
                    break
                    
            print("Here is the list of parameters you choose: ")
            print(parameters)
            correct = get_yes_no_input("Are they correct?")
            if correct == "yes":
                break
        return {
            "class": kls,
            "method": method,
            "parameters": parameters
        }