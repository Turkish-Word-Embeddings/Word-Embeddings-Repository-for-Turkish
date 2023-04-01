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

def scientific(number: float) -> str:
    factors = "{:.2E}".format(number).split("E")
    sign = factors[1][0]
    factors[1] = factors[1].lstrip("+-0")
    if sign == "-":
        factors[1] = "-"+factors[1]
    return f"{factors[0]} x $10^{{{factors[1]}}}$"

if __name__ == "__main__":

    args = get_experiment_args()
    folder_path = args.folder
    extension = args.extension

    metadata_attrs = ["pearson", "pearson-p", "spearman", "spearman-p", "oov-ratio"]

    folder_name = MetaData.get_folder_name(folder_path)
    results_file = folder_name + extension

    with MetaData(".", "txt", metadata_attrs, results_file) as _md:
        md = _md._metadata

    dump_file = open(f"{args.prepend_output}_{folder_name}.txt", "w", encoding="utf-8")

    lines = [
        r"\begin{table}[!htbp]"
        r"\centering",
        r"\scalebox{0.8}{",
        r"\begin{tabular}{|c|cc|}",
        r"\hline",
        r"Similarity Task                                             & \multicolumn{2}{c|}{Statistics}                  \\ \hline",
        r"\multicolumn{1}{|l|}{\multirow{3}{*}{Syntactic Similarity}} & \multicolumn{1}{c|}{Pearson Result: %.2f}  & p-value: %s \\ \cline{2-3} " % (md.loc["syntactic.txt", "pearson"], scientific(md.loc["syntactic.txt", "pearson-p"])),
        r"\multicolumn{1}{|l|}{}                                      & \multicolumn{1}{c|}{Spearman Result: %.2f} & p-value: %s \\ \cline{2-3} " % (md.loc["syntactic.txt", "spearman"], scientific(md.loc["syntactic.txt", "spearman-p"])),
        r"\multicolumn{1}{|l|}{}                                      & \multicolumn{2}{c|}{OOV Ratio: %.2f}                  \\ \hline" % (md.loc["syntactic.txt", "oov-ratio"]),
        r"\multirow{3}{*}{Semantic Similarity}                        & \multicolumn{1}{c|}{Pearson Result: %.2f}  & p-value: %s \\ \cline{2-3} " % (md.loc["semantic.txt", "pearson"], scientific(md.loc["semantic.txt", "pearson-p"])),
        r"& \multicolumn{1}{c|}{Spearman Result: %.2f} &  p-value: %s \\ \cline{2-3} " % (md.loc["semantic.txt", "spearman"], scientific(md.loc["semantic.txt", "spearman-p"])),
        r"& \multicolumn{2}{c|}{OOV Ratio: %.2f}                  \\ \hline" % (md.loc["semantic.txt", "oov-ratio"]),
        r"\end{tabular}}",
        r"\caption{\footnotesize{Placeholder}}",
        r"\end{table}",
    ]
    for line in lines:
        dump_file.write(f"{line}\n")