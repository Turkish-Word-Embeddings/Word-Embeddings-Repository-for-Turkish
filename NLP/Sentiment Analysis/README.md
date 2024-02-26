There are 3 different datasets used for Sentiment Analysis:
* **Turkish Movie Dataset**:  Türkmenoğlu C. and Tantuğ A.C. (2014). Sentiment analysis in Turkish media. In Proceedings of the Workshop on Issues of Sentiment Discovery and Opinion Mining, International Conference on Machine Learning, Beijing, China, pp. 1–11. 
* **Turkish Twitter Dataset**:  Aydın, C. R., & Güngör, T. (2020). Sentiment analysis in Turkish: Supervised, semi-supervised, and unsupervised techniques. Natural Language Engineering, 1-29. doi:10.1017/S1351324920000200
* **Turkish sentiment Analysis Dataset**:  Winvoker (2022). Turkish sentiment analysis dataset.URL: https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset


You can directly run `sentiment.py` to load the datasets "Turkish Movie Dataset", "Turkish Twitter Dataset" and "Turkish sentiment Analysis Dataset" (combination of product reviews and wiki in Turkish), and evaluate the performances of LSTM models trained using different word embedding methods. You will be prompted to specify which dataset you want to work with and which word embedding model you want to use. If not specified, it runs all the models.

You can also use `Turkish-Movie Sentiment Analysis.ipynb` to run the same process in a Jupyter notebook for "Turkish Movie Dataset".

Dataset `turkish-movie-dataset` within the `datasets` folder is the same as the "Turkish-Movie Dataset". It is just preprocessed and saved as a `.csv` file for easier access by `sentiment.py`.

Example runs:
```bash	
python sentiment.py --dataset 1 --embedding gl --model lstm  # Turkish Movie Dataset, GloVe word embedding, LSTM model
python sentiment.py -d 3 -hs 64 -e 10 -w w2v_sg -m lstm # Turkish sentiment Analysis Dataset, 64 hidden units, 10 epochs for Word2Vec
```