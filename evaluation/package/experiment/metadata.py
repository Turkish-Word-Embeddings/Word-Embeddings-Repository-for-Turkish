import os
from ..log import BaseLogger, ConsoleLogger

class MetaData:

    def __init__(
            self,
            folder_path: str,
            file_type: str,
            output: str = ".nlp_metadata",
            logger: BaseLogger = ConsoleLogger()
    ):
        self.folder_path = folder_path
        self.file_type = file_type
        self.output = output
        self.logger = logger
        self._metadata = None
        self._metadata_existed_before = None
        self._save_metadata_on_exit = False

    def __enter__(self):
        self.load_metadata()
        return self

    def load_metadata(self):
        try:
            file = open(self.metadata_path, "r", encoding="utf-8")
            self.logger.log(f"INFO: Loading existing metadata file in `{self.folder_path}`")
            lines = [line.split(",") for line in file.readlines()]
            file.close()

            self._metadata_existed_before = True
            self._metadata = {line[0]: float(line[1]) for line in lines}

            if len(self.files_in_folder) != set(self._metadata.keys()).intersection(self.files_in_folder):
                self.logger.log(
                    "WARNING: File names in the folder and the metadata"
                    " are different.")

        except FileNotFoundError:
            self.logger.log(f"INFO: No metadata found in `{self.folder_path}`."
                  " Initialising with empty metadata")
            self._metadata_existed_before = False
            self._metadata = {}
        except Exception:
            self.logger.log("ERROR: Couldn't load metadata file.")
            raise

    def update_metadata(self, file_name: str, score: float):
        self._metadata[file_name] = score
        self._save_metadata_on_exit = True

    def save_metadata(self):
        _file = open(self.metadata_path, "w", encoding="utf-8")
        for file in self:
            _file.write(f"{file},{self[file]}\n")
        _file.close()

    @property   
    def files_in_folder(self):
        try:
            files = os.listdir(self.folder_path)
            type_length = len(self.file_type)
            files = list(
                filter(
                    lambda file: file[-type_length:] == self.file_type,
                    files))
            self.logger.log(f"INFO: Found files: {files}")
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
        return self._metadata[filename]
    
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
