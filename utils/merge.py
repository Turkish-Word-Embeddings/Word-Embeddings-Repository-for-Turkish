class LineSentences(object):
    def __init__(self, filenames):
        self.filenames = filenames
    
    # memory-friendly iterator
    def __iter__(self):
        for filename in self.filenames:
            for line in open(filename, "r", encoding="utf-8"):
                yield line.strip().split()


FILE1 = "corpus/bounwebcorpus.txt"
FILE2 = "corpus/turkish-texts-tokenized.txt"
DEST = "corpus/merged.txt"

# iterate through files
sentences = LineSentences([FILE1, FILE2])
# print sentences to DEST
with open(DEST, "w", encoding="utf-8") as f:
    for sentence in sentences:
        f.write(" ".join(sentence) + "\n")
