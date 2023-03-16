import argparse
from package import add_metadata_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--folder",
        help="Folder to add metadata for.",
        required=True)
    parser.add_argument(
        "-ft",
        "--file-type",
        help="File types in the folder to add metadata for.",
        default="txt")
    parser.add_argument(
        "-e",
        "--extension",help="Metadata file extension",
        default=".nlp_metadata")
    args = parser.parse_args()

    add_metadata_file(
        args.folder,
        args.file_type,
        args.extension
    )