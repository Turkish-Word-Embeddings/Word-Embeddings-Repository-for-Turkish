# Word Embeddings Repository for Turkish

## What is this?

In this project, we aim at building a comprehensive word embedding [\[1\]](https://en.wikipedia.org/wiki/Word_embedding) repository for the Turkish language. Using each of the state-of-the-art word embedding methods, embeddings of all the words in the language will be formed using a corpus. First, the three commonly-used embedding methods (Word2Vec [\[2\]](https://arxiv.org/abs/1310.4546)-[\[3\]](https://arxiv.org/abs/1301.3781) , Glove [\[4\]](https://nlp.stanford.edu/pubs/glove.pdf), Fasttext [\[5\]](https://arxiv.org/abs/1607.04606)) will be used and an embedding dictionary for each one will be formed. Then we will continue with context-dependent embedding methods such as BERT [\[6\]](https://arxiv.org/abs/1810.04805) and Elmo [\[7\]](https://arxiv.org/abs/1802.05365). Each method will be applied with varying parameters such as different corpora and different embedding dimensions. The methods will be evaluated on analogy and similarity tasks.

## Quick Setup
* First of all, you need to clone the repository to your local machine. You can do this by typing the following command in your terminal:
```bash
git clone <url> <path>
```
* Create a virtual environment (optionally), and then install the requirements by typing the following command in your terminal:
```bash
python -m venv env  # Create a virtual environment called env
pip install -r requirements.txt
```
* You can run the `.py` scripts or `.ipynb` notebooks following the instructions. For details, please refer to our [wiki](https://github.com/Turkish-Word-Embeddings/Word-Embeddings-Repository-for-Turkish/wiki).


## Releases
* [Turkish Word Vectors, Corpus and Evaluation Dataset](https://github.com/Turkish-Word-Embeddings/Word-Embeddings-Repository-for-Turkish/releases/tag/v1.0.0)

## Open-source Turkish Corpora
* [BounWebCorpus](https://tulap.cmpe.boun.edu.tr/repository/xmlui/handle/20.500.12913/16)
* [HuaweiCorpus](https://github.com/onurgu/linguistic-features-in-turkish-word-representations/releases/tag/v1.0) - [\[8\]](https://www.cmpe.boun.edu.tr/~onurgu/publication/gungor-2017-linguistic/gungor-2017-linguistic.pdf)
* [Turkish CoNLL17 Corpus]( http://vectors.nlpl.eu/repository/) - [\[10\]]

## Task DataSet (Analogy & Similarity)
* [Analogy Test Set](https://github.com/onurgu/linguistic-features-in-turkish-word-representations/releases/tag/v1.0) - [\[8\]](https://www.cmpe.boun.edu.tr/~onurgu/publication/gungor-2017-linguistic/gungor-2017-linguistic.pdf)
* [Analogy/Similarity Test Set](https://github.com/bunyamink/word-embedding-models/tree/master/datasets)

## Requirements:

* `gensim==4.3.1` 
* `ipykernel==6.22.0` (for Jupyter notebooks - alternatively, you can use `.py` scripts)
* `py7zr==0.20.5` (for preprocessing)
* `pandas==2.0.0` (only for visualization of results)
* `matplotlib==3.7.1` (only for visualization of results)
* `tensorflow==2.10` (only for Word2Vec TensorFlow implementation)
* `elmoformanylangs==0.0.4` (only for Elmo)
    * Please install `overrides==3.1.0` if you face any problem with `elmoformanylangs`. Refer to the related [issue](https://github.com/HIT-SCIR/ELMoForManyLangs/issues/100).



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
10. Fares, M., Kutuzov, A., Oepen, S., & Velldal, E. (2017). Word vectors, reuse, and replicability: Towards a community repository of large-text resources. In Proceedings of the 21st Nordic Conference on Computational Linguistics (pp. 271-276). Association for Computational Linguistics. http://www.aclweb.org/anthology/W17-0237
11. Che, W., Liu, Y., Wang, Y., Zheng, B., & Liu, T. (2018). Towards better UD parsing: Deep contextualized word embeddings, ensemble, and treebank concatenation. In Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies (pp. 55-64). Association for Computational Linguistics. Brussels, Belgium. Retrieved from http://www.aclweb.org/anthology/K18-2005.
