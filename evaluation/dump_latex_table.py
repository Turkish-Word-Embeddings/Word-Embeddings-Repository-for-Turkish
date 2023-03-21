import argparse
from package import MetaData
from package.log import MockLogger

def get_experiment_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--folder",
        help="Folder of tasks to generate latex table for.",
        required=True
    )
    parser.add_argument(
        "-e",
        "--extension", help="Metadata file extension",
        default=".nlp_metadata"
    )
    parser.add_argument(
        "-dad",
        "--digits-after-decimal",
        default=4,
        type=int
    )
    parser.add_argument(
        "-po",
        "--prepend-output",
        default="latex_dump"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_experiment_args()
    folder_path = args.folder
    extension = args.extension

    number_of_examples = {}
    with MetaData(folder_path, "txt", ["mrr"], extension, MockLogger()) as md:
        original_metadata = md._metadata
        for file in md.files_in_folder:
            number_of_examples[file] = len(md.load_file(file))
    print(f"INFO: Found {original_metadata.shape[0]} files in original metadata")

    folder_name = MetaData.get_folder_name(folder_path)
    results_file = folder_name + extension
    with MetaData(".", "txt", ["mrr", "topn-miss-ratio"], results_file, MockLogger()) as md:
        result_metadata = md._metadata
    print(f"INFO: Found {result_metadata.shape[0]} files in results metadata")

    common_files = set(original_metadata.index).intersection(result_metadata.index)
    print(f"INFO: Common files: ", common_files)

    dump_file = open(f"{args.prepend_output}_{folder_name}.txt", "w", encoding="utf-8")

    # writing header
    dump_file.write("""
\\begin{table}[h]
\\centering
\\scalebox{0.7}{
\\begin{tabular}{|l|c|c|c|c|c|c|}
\\hline
Morphological Categories & \\multicolumn{1}{l|}{Number of examples} & \\multicolumn{1}{l|}{Topn Miss Ratio} & \\multicolumn{1}{l|}{MRR} & \\multicolumn{1}{l|}{Reference MRR} & \\multicolumn{1}{l|}{\\begin{tabular}[c]{@{}l@{}}Improvement with \\\\ respect to reference\\end{tabular}} \\\\ \\hline
    """)

    # writing files
    common_files = list(common_files)
    common_files.sort()
    improvement = lambda latest, reference: round(((latest / reference) - 1) * 100, 4)
    pdf = [[
        file,
        number_of_examples[file],
        result_metadata.loc[file, "topn-miss-ratio"],
        result_metadata.loc[file, "mrr"],
        original_metadata.loc[file, "mrr"],
        improvement(result_metadata.loc[file, "mrr"], original_metadata.loc[file, "mrr"])
    ] for file in common_files]
    for row in pdf:
        dump_file.write(" & ".join([str(i) for i in row]))
        dump_file.write("\\% \\\\ \\hline\n")

    # writing overall
    acc = [0, 0, 0, 0]
    for row in pdf:
        acc[0] += row[1]
        for i in range(1, 4):
            # import pdb; pdb.set_trace()
            acc[i] += row[i + 1] * row[1]
    for i in range(1, 4):
        acc[i] /= acc[0]
    acc.append(improvement(acc[2], acc[3]))
    dump_file.write(r"\rowcolor{yellow}\multicolumn{1}{|c|}{Overall} & ")
    dump_file.write(" & ".join([str(round(i, 4)) for i in acc]))
    dump_file.write(r"\%\\ \hline")

    # writing footer
    dump_file.write("""

\\end{tabular}}
\\caption{\\footnotesize{\\textit{PLACE}: HOLDER}}
\\end{table}
    """)

    dump_file.close()