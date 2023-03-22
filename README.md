# Word Embeddings Repository for Turkish

## What is this?

In this project, we aim at building a comprehensive word embedding [\[1\]](https://en.wikipedia.org/wiki/Word_embedding) repository for the Turkish language. Using each of the state-of-the-art word embedding methods, embeddings of all the words in the language will be formed using a corpus. First, the three commonly-used embedding methods (Word2Vec [\[2\]](https://arxiv.org/abs/1310.4546)-[\[3\]](https://arxiv.org/abs/1301.3781) , Glove [\[4\]](https://nlp.stanford.edu/pubs/glove.pdf), Fasttext [\[5\]](https://arxiv.org/abs/1607.04606)) will be used and an embedding dictionary for each one will be formed. Then we will continue with context-dependent embedding methods such as BERT [\[6\]](https://arxiv.org/abs/1810.04805) and Elmo [\[7\]](https://arxiv.org/abs/1802.05365). Each method will be applied with varying parameters such as different corpora and different embedding dimensions. The methods will be evaluated on analogy and similarity tasks.

## How to use?
* First of all, you need to clone the repository to your local machine. You can do this by typing the following command in your terminal:
```bash
git clone <url> <path>
```
* Create a virtual environment (optionally), and then install the requirements by typing the following command in your terminal:
```bash
python -m venv env  # Create a virtual environment called env
pip install -r requirements.txt
```

### Preprocess
* To learn the embedding vectors for Turkish words, we have to use a corpus. Put your corpus file into the working directory. 
* If they are `7-zip` files, you can first use the corresponding script to convert them into `txt` file. For example, to convert `wiki.tr.txt.7z` file, use the following command. (`--output` indicates the output folder for your `txt` file to be stored. If you do not specify it, the output file will be stored in the working directory.):
    ```bash
    python preprocess/7z_to_txt.py --input wiki.tr.txt.7z --output wiki.tr.txt
    ```
* If the `txt` version includes redundant lines, you can format your `txt` file using the `txt_formatter` script, which basically re-creates the file using the provided _stride_ and _offset_ values. For example, to format the `wiki.tr.txt.7z` file, you can use the following command:
    ```bash
    python preprocess/txt_formatter.py -i wiki.tr.txt.7z -s 4 -f 1 
    ```
    If not provided, `--output` defaults to the input file. `--stride` and `--offset` default to 1 and 0, respectively. _stride_ stands for the number of lines to skip between consecutive sentences and _offset_ stands for the number of lines to skip at the beginning of the file.
* Additionally, you can use the `analyzer` script to get vocabulary size and maximum sequence length in your corpus. For example, to get the vocabulary size and maximum sequence length of the `wiki.tr.txt.7z` file, you can use the following command:
    ```bash
    python preprocess/txt_analyzer.py -i wiki.tr.txt.7z
    ```

## Models
* [TensorFlow Word2Vec (Skip-Gram with Negative Sampling)](https://github.com/Turkish-Word-Embeddings/Word-Embeddings-Repository-for-Turkish/blob/main/word2vec/tf_w2v.ipynb) 
* [Gensim Word2Vec (Skip-Gram/CBOW with Negative Sampling/Hierarchical Softmax)](https://github.com/Turkish-Word-Embeddings/Word-Embeddings-Repository-for-Turkish/blob/main/word2vec/gensim_w2v.ipynb) 
* [FastText (Skip-Gram/CBOW with Softmax/Negative Sampling/Hierarchical Softmax/One-vs-All)](https://github.com/Turkish-Word-Embeddings/Word-Embeddings-Repository-for-Turkish/blob/main/fasttext/fasttext.ipynb)


## Open-source Turkish Corpora
* [TULAP Turkish-English Parallel Corpus](https://tulap.cmpe.boun.edu.tr/repository/xmlui/handle/20.500.12913/19)
* [BounWebCorpus](https://tulap.cmpe.boun.edu.tr/repository/xmlui/handle/20.500.12913/16)
* [HuaweiCorpus](https://github.com/onurgu/linguistic-features-in-turkish-word-representations/releases/tag/v1.0) - [\[8\]](https://www.cmpe.boun.edu.tr/~onurgu/publication/gungor-2017-linguistic/gungor-2017-linguistic.pdf)

## Task DataSet (Analogy & Similarity)
* [Analogy Test Set](https://github.com/onurgu/linguistic-features-in-turkish-word-representations/releases/tag/v1.0) - [\[8\]](https://www.cmpe.boun.edu.tr/~onurgu/publication/gungor-2017-linguistic/gungor-2017-linguistic.pdf)
* [Analogy/Similarity Test Set](https://github.com/bunyamink/word-embedding-models/tree/master/datasets)

## References
1. https://en.wikipedia.org/wiki/Word_embedding
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., Dean, J. (2013). Distributed representations of words and phrases and their compositionality. arXiv preprint arXiv:1310.4546.
3. Mikolov, T., Chen, K., Corrado, G., Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.
4. Pennington, J., Socher, R., Manning, C. D. (2014). Glove: Global vectors for word representation. In Proc. of the Conference on Empirical Methods in Natural Language Processing (EMNLP), p.1532-1543.
5. Bojanowski, P., Grave, E., Joulin, A., Mikolov, T. (2017). Enriching word vectors with subword information. Transactions of the Association for Computational Linguistics, Vol.5, p.135-146.
6. Devlin, J., Chang, M. W., Lee, K., Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
7. Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., Zettlemoyer, L. (2018). Deep contextualized word representations. arXiv preprint arXiv:1802.05365.
8. Onur Gungor, Eray Yildiz, "Linguistic Features in Turkish Word Representations - Türkçe Sözcük Temsillerinde Dilbilimsel Özellikler", 2017 25th Signal Processing and Communications Applications Conference (SIU), Antalya, 2017.
9. Grave, E., Bojanowski, P., Gupta, P., Joulin, A., & Mikolov, T. (2018). Learning word vectors for 157 languages. In Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018).
