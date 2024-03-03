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
* [Boun Web Corpus](https://tulap.cmpe.boun.edu.tr/entities/corpus/c9f404aa-64da-4be5-a173-99b141bde7bd)
* [HuaweiCorpus](https://github.com/onurgu/linguistic-features-in-turkish-word-representations/releases/tag/v1.0) [\[8\]](https://www.cmpe.boun.edu.tr/~onurgu/publication/gungor-2017-linguistic/gungor-2017-linguistic.pdf)
* [Turkish CoNLL17 Corpus]( http://vectors.nlpl.eu/repository/)

## Datasets for Intrinsic Evaluation

| Dataset                                                      | Category            | \# Instances |
|:--------------------------------------------------------------:|:---------------------:|:--------------:|
| Dataset by Güngör *et al.* [\[12\]](https://ieeexplore.ieee.org/document/7960223/)                            | Syntactic Analogy  | 29,364       |
| Dataset by Kurt [\[13\]](https://github.com/bunyamink/word-embedding-models/tree/master/datasets/analogy)                                            | Semantic Analogy   | 3,296        |
| *WordSimTr* [\[14\]](https://wlv.openrepository.com/handle/2436/623576)                                                 | Syntactic Similarity | 140          |
| *AnlamVer* [\[15\]](https://aclanthology.org/C18-1323/)                                                  | Semantic Similarity | 500          |


## Datasets for Extrinsic Evaluation

| Task                   | Dataset                                      | Train (\# sentences) | Test (\# sentences) |
|:------------------------:|:----------------------------------------------:|:----------------------:|:---------------------:|
| Sentiment analysis     | Turkish Movie Dataset  [\[16\]](https://www.researchgate.net/publication/269634534_Sentiment_Analysis_in_Turkish_Media)                 | 16,100               | 4,144               |
| Sentiment analysis     | Turkish Sentiment Analysis Dataset  [\[17\]](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset)      | 286,854              | 32,873              |
| Sentiment analysis     | Turkish Twitter Dataset  [\[18\]](https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/sentiment-analysis-in-turkish-supervised-semisupervised-and-unsupervised-techniques/3E5CAB8E6A2B8877135F63485536C8F9)      | 1,055           | 476            |
| Named entity recognition | Turkish National Newspapers with NER labels  [\[19\]](https://aclanthology.org/C18-1177/)  | 28,468               | 2,915               |
| PoS tagging            | UD BOUN Treebank  [\[20\]](https://arxiv.org/abs/2002.10416)                       | 8,782                | 979                 |

*Within the context of intrinsic evaluation, the employed datasets have been compiled and made accessible herein, with the aim of facilitating reproducibility and enabling their integration into further investigations by researchers. With regard to the dataset presented by Güngör et al., we partitioned it into designated subcategories and subsequently arranged and preserved these divisions within this repository. Similarity scores in *WordSimTr* are normalized to a numeric scale ranging from 0 to 10 for compatibility with other similarity pairs.*


## Requirements:

* `gensim==4.3.1` 
* `ipykernel==6.22.0` (for Jupyter notebooks - alternatively, you can use `.py` scripts)
* `py7zr==0.20.5` (for preprocessing)
* `pandas==2.0.0` (only for visualization of results)
* `matplotlib==3.7.1` (only for visualization of results)
* `tensorflow==2.10` (only for Word2Vec TensorFlow implementation)
* `elmoformanylangs==0.0.4` (only for Elmo)
    * Please install `overrides==3.1.0` if you face any problem with `elmoformanylangs`. Refer to the related [issue](https://github.com/HIT-SCIR/ELMoForManyLangs/issues/100).

## Reproducibility
* To reproduce the intrinsic evaluation results, you should download the corresponding word embedding model from the [release section](https://github.com/Turkish-Word-Embeddings/Word-Embeddings-Repository-for-Turkish/releases/tag/v1.0.0) and then run the necessary scripts in the [evaluation](https://github.com/Turkish-Word-Embeddings/Word-Embeddings-Repository-for-Turkish/tree/main/evaluation) folder following the instructions provided [here](https://github.com/Turkish-Word-Embeddings/Word-Embeddings-Repository-for-Turkish/tree/main/evaluation#readme).
* To reproduce the extrinsic evaluation results, you should download the corresponding word embedding model from the [release section](https://github.com/Turkish-Word-Embeddings/Word-Embeddings-Repository-for-Turkish/releases/tag/v1.0.0), then run the necessary NLP task in the [NLP](https://github.com/Turkish-Word-Embeddings/Word-Embeddings-Repository-for-Turkish/tree/main/NLP) folder with random seeds 7, 24, and 30 (for the third Sentiment Analysis task, please provide hidden size as 196: `-hs 196`). Results provided in our paper are averaged results over these runs with Wilson Intervals.
## Citation
For now, you can cite the preprint version of our study. 
```
Sarıtaş, Karahan and Oz, Cahid Arda and Güngör, Tunga,
A Comprehensive Analysis of Static Word Embeddings for Turkish.
Available at SSRN: https://ssrn.com/abstract=4615519 or http://dx.doi.org/10.2139/ssrn.4615519
```
$\text{BibTeX}$ citation:
```
@article{2023sweturkish,
  title = {A Comprehensive Analysis of Static Word Embeddings for Turkish},
  author = {Sarıtaş, Karahan and Oz, Cahid Arda and Güngör, Tunga},
  year = {2023},
  note = {Available at SSRN: \url{https://ssrn.com/abstract=4615519} or DOI: \url{http://dx.doi.org/10.2139/ssrn.4615519}}
}
```

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
12. Onur Güngör and Eray Yıldız. 2017. Linguistic features in Turkish word representations. In 2017 25th Signal Processing and Communications
Applications Conference (SIU). 1–4. https://doi.org/10.1109/SIU.2017.7960223
13. Bünyamin Kurt. 2018. Word Embedding Models - Datasets. https://github.com/bunyamink/word-embedding-models/tree/master/datasets/analogy
14. Aylin Üstün, Murat Kurfalı, and Burcu Can. 2018. Characters or morphemes: How to represent words?. In Proceedings of The Third
Workshop on Representation Learning for NLP. 144–153.
15. Gökhan Ercan and Olcay Taner Yıldız. 2018. AnlamVer: Semantic Model Evaluation Dataset for Turkish - Word Similarity and Relatedness.
In Proceedings of the 27th International Conference on Computational Linguistics. Association for Computational Linguistics, Santa Fe,
New Mexico, USA, 3819–3836. https://aclanthology.org/C18-1323/
16. Cumali Türkmenoğlu and A. Cüneyd Tantuğ. 2014. Sentiment Analysis in Turkish Media. In Workshop on Issues of Sentiment Discovery and Opinion Mining, International Conference on Machine Learning (ICML).
17. Winvoker. 2022. Turkish Sentiment Analysis Dataset. [https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset)
18. Aydın, C. R., & Güngör, T. (2021). Sentiment analysis in Turkish: Supervised, semi-supervised, and unsupervised techniques. Natural Language Engineering, 27(4), 455–483. doi:10.1017/S1351324920000200
19. Onur Gungor, Suzan Uskudarli, and Tunga Gungor. 2018. Improving Named Entity Recognition by Jointly Learning to Disambiguate Morphological Tags. In Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018).
20. Utku Türk, Furkan Atmaca, Saziye Betül Özates, Gözde Berk, Seyyit Talha Bedir, Abdullatif Köksal, Balkiz Öztürk Basaran, Tunga Güngör, and Arzucan Özgür. 2020. Resources for Turkish Dependency Parsing: Introducing the BOUN Treebank and the BoAT Annotation Tool. CoRR abs/2002.10416 (2020). arXiv:2002.10416 https://arxiv.org/abs/2002.10416
