import argparse
from package import load_model
from package import evaluate_analogy_folder, evaluate_similarity_folder

def get_experiment_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Model to evaluate",
        required=True
    )
    parser.add_argument(
        "-af",
        "--analogy-folder",
        action="append",
        help="Folder of analogy tasks to evaluate."
    )
    parser.add_argument(
        "-sf",
        "--similarity-folder",
        action="append",
        help="Folder of similarity tasks to evaluate."
    )
    parser.add_argument(
        "-topn",
        "--topn",
        help="topn value to use",
        default=10
    )
    parser.add_argument(
        "-ft",
        "--file-type",
        help="File types in the folder to add metadata for.",
        default="txt")
    parser.add_argument(
        "-o",
        "--output",help="Name of the metadata file to write",
        default="experiment")
    parser.add_argument(
        "-e",
        "--extension",help="Metadata file extension",
        default=".nlp_metadata")
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbosity",
        default=1,
        type=int)
    args = parser.parse_args()

    if args.analogy_folder is None and args.similarity_folder is None:
        raise ValueError("You must pass at least one -af or -sf argument. You passed none.")

    return args

if __name__ == '__main__':
    args = get_experiment_args()
    model = load_model(args.model, args.extension)

    for analogy_folder in args.analogy_folder or []:
        evaluate_analogy_folder(
            model,
            args.topn,
            analogy_folder,
            args.file_type,
            args.extension,
            args.verbose
        )

    for similarity_folder in args.similarity_folder or []:
        evaluate_similarity_folder(
            model,
            similarity_folder,
            args.file_type,
            args.extension,
            args.verbose
        )