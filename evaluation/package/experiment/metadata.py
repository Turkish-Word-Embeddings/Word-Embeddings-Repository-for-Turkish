import pandas as pd
from typing import Optional, List, Dict
import os
from ..log import BaseLogger, ConsoleLogger

class MetaData:

    def __init__(
            self,
            folder_path: str,
            file_type: str,
            metadata_attrs: List[str],
            output: str = ".nlp_metadata",
            logger: BaseLogger = ConsoleLogger(),
    ):
        self.folder_path = folder_path
        self.file_type = file_type
        self.output = output
        self.logger = logger
        self.metadata_attrs = metadata_attrs
        self._metadata: Optional[pd.DataFrame] = None
        self._metadata_existed_before = None
        self._files_in_folder = None
        self._save_metadata_on_exit = False

    def __enter__(self):
        self.load_metadata()
        return self

    def load_metadata(self):
        try:
            self.logger.log(f"INFO: Loading existing metadata file in `{self.folder_path}`")
            self._metadata_existed_before = True
            self._metadata = pd.read_csv(self.metadata_path, encoding="utf-8")
            self._metadata = self._metadata.round(4)
            self._metadata.set_index("filename", inplace=True)

            if len(self._metadata.columns) != len(self.metadata_attrs):
                raise ValueError(
                    "Number of attributes of the metadata and the file"
                    " do not match. Please reduce number of attributes"
                    " or delete the current file.\n"
                    f" - Attributes in the file: {list(self._metadata.columns)}\n"
                    f" - Expected: {self.metadata_attrs}")

            if len(self.files_in_folder) != set(self._metadata.index).intersection(self.files_in_folder):
                self.logger.log(
                    "WARNING: File names in the folder and the metadata"
                    " are different.")

        except FileNotFoundError:
            self.logger.log(f"INFO: No metadata found in `{self.folder_path}`."
                  " Initialising with empty metadata")
            self._metadata_existed_before = False
            self._metadata: pd.DataFrame = pd.DataFrame(columns=self.metadata_attrs)
        except Exception:
            self.logger.log("ERROR: Couldn't load metadata file.")
            raise

    def update_metadata(self, file_name: str, scores: Dict[str, float]):
        self._metadata.loc[file_name] = [scores[metric] for metric in self.metadata_attrs]
        self._save_metadata_on_exit = True

    def save_metadata(self):
        self._metadata.index.name = "filename"
        self._metadata.to_csv(self.metadata_path, index=True)

    @property   
    def files_in_folder(self):

        if self._files_in_folder != None:
            return self._files_in_folder
        try:
            files = os.listdir(self.folder_path)
            type_length = len(self.file_type)
            files = list(
                filter(
                    lambda file: file[-type_length:] == self.file_type,
                    files))
            self.logger.log(f"INFO: Found files: {files}")
            self._files_in_folder = files
            return files

        except FileNotFoundError:
            raise

    def load_file(self, file_name):
        file_path = os.path.join(self.folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.split() for line in f.readlines()]

    def __exit__(self, type, value, traceback):
        if type:
            raise
        
        if self._save_metadata_on_exit:
            self.save_metadata()
            self.logger.log(f"INFO: Metadata of `{self.folder_path}` is updated.")
        else:
            self.logger.log("INFO: No changes made to metadata, exiting without changing"
                  " the metadata file.")

    def __getitem__(self, filename):
        return self._metadata.loc[filename, :]
    
    def __iter__(self):
        files = list(self._metadata.keys())
        files.sort()
        for file in files:
            yield file

    @property
    def metadata_exists(self):
        "returns true if metadata already exists in folder"
        return self._metadata_existed_before
    
    @property
    def metadata_path(self):
        return os.path.join(self.folder_path, self.output)

    @classmethod
    def get_folder_name(cls, folder_path):
        return os.path.normpath(folder_path).split(os.path.sep)[-1]
