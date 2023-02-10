#####
# This script formats a txt file using provided stride and offset values. It also gets rid of punctuation.
#####
import string
import argparse

LOWER = {ord(u'A'): u'a',
    ord(u'A'): u'a',
    ord(u'B'): u'b',
    ord(u'C'): u'c',
    ord(u'Ç'): u'ç',
    ord(u'D'): u'd',
    ord(u'E'): u'e',
    ord(u'F'): u'f',
    ord(u'G'): u'g',
    ord(u'Ğ'): u'ğ',
    ord(u'H'): u'h',
    ord(u'I'): u'ı',
    ord(u'İ'): u'i',
    ord(u'J'): u'j',
    ord(u'K'): u'k',
    ord(u'L'): u'l',
    ord(u'M'): u'm',
    ord(u'N'): u'n',
    ord(u'O'): u'o',
    ord(u'Ö'): u'ö',
    ord(u'P'): u'p',
    ord(u'R'): u'r',
    ord(u'S'): u's',
    ord(u'Ş'): u'ş',
    ord(u'T'): u't',
    ord(u'U'): u'u',
    ord(u'Ü'): u'ü',
    ord(u'V'): u'v',
    ord(u'Y'): u'y',
    ord(u'Z'): u'z'}



# example usage: python preprocess/txt_formatter.py -i "corpus/Turkish-English Parallel Corpus.txt" -s 4 -f 1  
if __name__ == '__main__':
    # take input corpora file, batch size, embedding dimensions and output file name as command line arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",  help="Input txt file. If file name includes space(s), enclose it in double quotes.", required=True)
    parser.add_argument("-o", "--output", help="Output file. If file name includes space(s), enclose it in double quotes. If not provided, overwrites the input file.",)
    parser.add_argument("-s", "--stride", help="Number of lines to skip between two sentences.", default=1, type=int)
    parser.add_argument("-f", "--offset", help="Number of lines to skip from the beginning of the file.", default=0, type=int)
    args = parser.parse_args()

    input = args.input
    output = args.output if args.output else input
    stride = args.stride
    offset = args.offset

    with open(input, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sentences = [lines[i].strip() for i in range(offset, len(lines), stride)]
    # write sentences to output file
    with open(output, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence.translate(LOWER).strip().translate(str.maketrans('', '', string.punctuation)) + "\n")

    
    