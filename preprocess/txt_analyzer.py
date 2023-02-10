#####
# This script analyzes a txt file.
#####

import argparse
from functools import reduce
 
# Assumption: Input file is assumed to have lowercase, non-punctuated sentences. If not, you can use scripts\txt_formatter.py to format the file.
# example usage: python preprocess/txt_analyzer.py -i "corpus/Turkish-English Parallel Corpus.txt"
if __name__ == '__main__':
    # take input corpora file, batch size, embedding dimensions and output file name as command line arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",  help="Input txt file. If file name includes space(s), enclose it in double quotes.", required=True)
    args = parser.parse_args()

    input = args.input
    # read input file, get rid of puntuation and split into sentences
    with open(input, "r", encoding="utf-8") as f:
        lines = [line.split() for line in f.readlines()]

    max_line, max_len = reduce(lambda acc, line: (line, len(line)) if len(line) > acc[1] else acc, lines, ('', 0))
    # find number of different words
    words = set()
    for line in lines:
        words.update(line)
    

    print("Number of sentences: ", len(lines))
    print("Number of unique words (vocabulary size): ", len(words))
    print("Maximum length of a sentence: ", max_len)
    # print("Sentence with maximum length: ", max_line)


    
    