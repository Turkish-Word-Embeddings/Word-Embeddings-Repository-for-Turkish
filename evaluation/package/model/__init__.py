from .model_metadata import Model_MetaData

def load_model(model_path: str, extension: str):
    model_metadata = Model_MetaData(model_path, extension)
    return model_metadata.load_model()