#####
# This script takes a 7z file as input and extracts the text files from it.
#####

# Example usage:  (put 7-zip files into "corpus" folder)
# python preprocess/7zip_to_txt.py -i corpus/Turkish-English_Parallel_Corpus.7z.001 corpus/Turkish-English_Parallel_Corpus.7z.002 -o corpus
import py7zr
import os
import argparse

if __name__ == '__main__':
    # take input corpora file, batch size, embedding dimensions and output file name as command line arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", nargs='+', help="Input corpora file(s). If file name includes space(s), enclose it in double quotes.", required=True)
    parser.add_argument("-o", "--output", help="Output folder. If file name includes space(s), enclose it in double quotes.")
    args = parser.parse_args()

    filenames = [file.strip() for file in args.input]
    with open('corpus/result.7z', 'ab') as outfile:  # append in binary mode
        for fname in filenames:
            with open(fname, 'rb') as infile:        # open in binary mode also
                outfile.write(infile.read())
    with py7zr.SevenZipFile("corpus/result.7z", "r") as archive:
        archive.extractall(args.output)  
    os.unlink("corpus/result.7z")  # delete the 7z file