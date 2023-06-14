import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn import model_selection
from sklearn import metrics
import torch
import torch.nn as nn
import tensorflow as tf  # pytorch for the model, tensorflow for tokenizer
import gensim
from dataset import Dataset
import argparse
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchmetrics.classification import BinaryF1Score

FOLDER = "C:/Users/karab/Desktop/Models"  # change this according to your folder structure
CONFIG = {
        "w2v_sg": {"model": os.path.join(FOLDER, "word2vec_10ep-300emb.bin"), "dim": 300, "binary": True, "no_header": False},
        "w2v_cbow": {"model": os.path.join(FOLDER, "word2vec-cbow-10ep-300emb.bin"), "dim": 300, "binary": True, "no_header": False},
        "ft_sg": {"model": os.path.join(FOLDER, "fasttext-10ep-300emb.bin"), "dim": 300, "binary": True, "no_header": False},
        "w2v_ft_sg_avg": {"model": os.path.join(FOLDER, "word2vec-fasttext-average.bin"), "dim": 300, "binary": True, "no_header": False},
        "glove": {"model": os.path.join(FOLDER, "glove.txt"), "dim": 300, "binary": False, "no_header": True},
        "dc_elmo": {"model": os.path.join(FOLDER, "elmo-decontextualized-static.bin"), "dim": 1024, "binary": True, "no_header": False},
        "dc_bert": {"model": os.path.join(FOLDER, "bert-decontextualized-static.wv"), "dim": 768, "binary": False, "no_header": False},
        "x2_bert": {"model": os.path.join(FOLDER, ""), "dim": 768, "binary": False, "no_header": False}
        }

class Config:
    def __init__(self, model, max_len, train_batch_size, valid_batch_size, epochs):
        self.model = model
        self.max_len = max_len
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.epochs = epochs
        self.dim = CONFIG[self.model]["dim"]
        self.word_vectors = self.load_wv()
    
    def load_wv(self):
        print('loading word embeddings...')
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(
            CONFIG[self.model]["model"],
            binary=CONFIG[self.model]["binary"],
            no_header=CONFIG[self.model]["no_header"]
        )
        return word_vectors

def preprocess(dataset):
    # read train and test csv
    if(dataset == 1): folder = "data1"
    else: folder = "data2"
    traindf = pd.read_csv(os.path.join(folder, "train.csv"))
    testdf = pd.read_csv(os.path.join(folder, "test.csv"))

    # shuffle
    traindf = traindf.sample(frac=1).reset_index(drop=True)
    testdf = testdf.sample(frac=1).reset_index(drop=True)

    # get label
    trainy = traindf['sentiment'].values
    testy = testdf['sentiment'].values

    negatives = traindf[traindf['sentiment'] == 0].shape[0]
    positives = traindf[traindf['sentiment'] == 1].shape[0]

    return traindf, testdf, trainy, testy, negatives, positives

def training(data_loader, model, optimizer, device):
    """
    this is model training for one epoch
    data_loader:  this is torch dataloader, just like dataset but in torch and devide into batches
    model : lstm
    optimizer : torch optimizer : adam
    device:  cuda or cpu
    """
    # set model to training mode
    model.train()
    # go through batches of data in data loader
    for data in data_loader:
        reviews = data['review']
        targets = data['target']
        # move the data to device that we want to use
        reviews = reviews.to(device, dtype = torch.long)
        targets = targets.to(device, dtype = torch.float)
        # clear the gradient
        optimizer.zero_grad()
        # make prediction from model
        predictions = model(reviews)
        # caculate the losses
        loss = nn.BCEWithLogitsLoss()(predictions, targets.view(-1,1))
        # backprob
        loss.backward()
        #single optimization step
        optimizer.step()

def evaluate(data_loader, model, device):
    final_predictions = []
    final_targets = []
    model.eval()
    # turn off gradient calculation
    with torch.no_grad():
        for data in data_loader:
            reviews = data['review']
            targets = data['target']
            reviews = reviews.to(device, dtype = torch.long)
            targets = targets.to(device, dtype = torch.float)
            # make prediction
            predictions = model(reviews)
            # move prediction and target to cpu
            predictions = predictions.cpu().numpy().tolist()
            targets = targets.cpu().numpy().tolist()
            # add predictions to final_prediction
            final_predictions.extend(predictions)
            final_targets.extend(targets)
    return final_predictions, final_targets

class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        super(LSTM, self).__init__()
        # Number of words = number of rows in embedding matrix
        num_words = embedding_matrix.shape[0]
        # Dimension of embedding is num of columns in the matrix
        embedding_dim = embedding_matrix.shape[1]
        # Define an input embedding layer
        self.embedding = nn.Embedding(num_embeddings=num_words,
                                      embedding_dim=embedding_dim)
        # Embedding matrix actually is collection of parameter
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype = torch.float32))
        # Because we use pretrained embedding (GLove, Fastext,etc) so we turn off requires_grad-meaning we do not train gradient on embedding weight
        self.embedding.weight.requires_grad = False
        # LSTM with hidden_size = 16
        self.lstm = nn.LSTM(
                            embedding_dim, 
                            16,
                            bidirectional=False,
                            batch_first=True,
                             )
        self.out = nn.Linear(32, 1)
    def forward(self, x):
        # pass input (tokens) through embedding layer
        x = self.embedding(x)
        # fit embedding to LSTM
        hidden, _ = self.lstm(x)
        # apply mean and max pooling on lstm output
        avg_pool= torch.mean(hidden, 1)
        max_pool, index_max_pool = torch.max(hidden, 1)
        # concat avg_pool and max_pool (so we have 256 size, also because this is bidirectional ==> 256*2 = 512)
        out = torch.cat((avg_pool, max_pool), 1)
        # fit out to self.out to conduct dimensionality reduction from 512 to 1
        out = self.out(out)
        # return output
        return out

def create_embedding_matrix(word_index, embedding_dict=None, dim=300):
    """
     this function create the embedding matrix save in numpy array
    :param word_index: a dictionary with word: index_value
    :param embedding_dict: a dict with word embedding
    :d_model: the dimension of word pretrained embedding
    :return a numpy array with embedding vectors for all known words
    """
    embedding_matrix = np.zeros((len(word_index) + 1, dim))
    ## loop over all the words
    for word, index in word_index.items():
        if word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]
    return embedding_matrix

# embedding_dict['word'] = vector
# word_index['word'] = index
# embedding_matrix[index] = vector
# index comes from tokenizer, sentences will be in the form of [index1, index2, ..., indexn]

def tokenize(traindf):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(traindf['review'].values.tolist())
    return tokenizer

def train_lstm(tokenizer, wv, dim, traindf, max_len, train_bs, valid_bs, epochs):
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict=wv, dim=dim)
    
    # split traindf %80 %20
    train_df, valid_df = model_selection.train_test_split(traindf, test_size=0.2)
    # STEP 3: pad sequence
    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)
    
    # zero padding
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=max_len)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=max_len)
    
    # STEP 4: initialize dataset class for training
    train_dataset = Dataset(reviews=xtrain, targets=train_df['sentiment'].values)
    
    # STEP 5: Load dataset to Pytorch DataLoader
    # after we have train_dataset, we create a torch dataloader to load train_dataset class based on specified batch_size
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = train_bs, num_workers=2)
    valid_dataset = Dataset(reviews=xtest, targets=valid_df['sentiment'].values)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = valid_bs, num_workers=1)
    
    # STEP 6: Running 
    device = torch.device('cuda')
    model = LSTM(embedding_matrix)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print('training model')
    for epoch in tqdm(range(epochs)):
        training(train_data_loader, model, optimizer, device)
        outputs, targets = evaluate(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f'epoch: {epoch}, Validation Accuracy Score: {accuracy}')
    return model


def test_lstm(tokenizer, maxlen, testdf, testy, model):
    device = torch.device('cuda')
    xtest = tokenizer.texts_to_sequences(testdf.review.values)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=maxlen)
    test_dataset = Dataset(reviews=xtest, targets=testy)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = valid_bs, num_workers=1)
    outputs, targets = evaluate(test_data_loader, model, device)
    outputs = np.array(outputs) >= 0.5
    acc = metrics.accuracy_score(targets, outputs)

    metric = BinaryF1Score()
    f1 = metric(torch.tensor(targets).reshape(-1, 1), torch.tensor(outputs).reshape(-1, 1))
    return acc, f1 
    

def train_lr(tokenizer, wv, dim, traindf, testdf, max_len):
    # apply logistic regression
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict=wv, dim=dim)
    
    # STEP 3: pad sequence
    xtrain = tokenizer.texts_to_sequences(traindf.review.values)
    xtest = tokenizer.texts_to_sequences(testdf.review.values)
    
    # zero padding
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=max_len)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=max_len)    

    # apply logistic regression using embedding matrix
    
    # xtrain.shape = # of training reviews, maxlen
    # xtest.shape = # of training reviews, maxlen 
    # embedding_matrix.shape = # of words, dim

    # our input --> # of training reviews, maxlen, dim
    # our output --> # of training reviews, label
    
    xtrain_embedded = np.zeros((len(xtrain), dim))
    xtest_embedded = np.zeros((len(xtest), dim))
    
    for i in range(len(xtrain)):
        xtrain_embedded[i] = np.mean(np.array([embedding_matrix[x] for x in xtrain[i]]), axis=0)
    for i in range(len(xtest)):
        xtest_embedded[i] = np.mean(np.array([embedding_matrix[x] for x in xtest[i]]), axis=0)

    lr = LogisticRegression()
    lr.fit(xtrain_embedded, traindf['sentiment'].values)
    preds = lr.predict(xtest_embedded)
    acc = metrics.accuracy_score(testdf['sentiment'].values, preds)
    
    metric = BinaryF1Score()
    f1 = metric(torch.tensor(testdf['sentiment'].values).reshape(-1, 1), torch.tensor(preds).reshape(-1, 1))
    return acc, f1


def train_svm(tokenizer, wv, dim, traindf, testdf, max_len):
    # apply logistic regression
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict=wv, dim=dim)

    # STEP 3: pad sequence
    xtrain = tokenizer.texts_to_sequences(traindf.review.values)
    xtest = tokenizer.texts_to_sequences(testdf.review.values)
    
    # zero padding
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=max_len)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=max_len)    

    # apply logistic regression using embedding matrix
    
    # xtrain.shape = # of training reviews, maxlen
    # xtest.shape = # of training reviews, maxlen 
    # embedding_matrix.shape = # of words, dim

    # our input --> # of training reviews, maxlen, dim
    # our output --> # of training reviews, label
    
    xtrain_embedded = np.zeros((len(xtrain), dim))
    xtest_embedded = np.zeros((len(xtest), dim))
    
    for i in range(len(xtrain)):
        xtrain_embedded[i] = np.mean(np.array([embedding_matrix[x] for x in xtrain[i]]), axis=0)
    for i in range(len(xtest)):
        xtest_embedded[i] = np.mean(np.array([embedding_matrix[x] for x in xtest[i]]), axis=0)

    # apply svm
    # 3000: 0.677
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', max_iter = 30000))
    print("fitting starts")
    clf.fit(xtrain_embedded, traindf['sentiment'].values)
    print("fitting completed")
    preds = clf.predict(xtest_embedded)
    acc = metrics.accuracy_score(testdf['sentiment'].values, preds)
    metric = BinaryF1Score()
    f1 = metric(torch.tensor(testdf['sentiment'].values).reshape(-1, 1), torch.tensor(preds).reshape(-1, 1))

    return acc, f1 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--word_embedding", help="Word embedding file to experiment on.", choices=["w2v_sg", "w2v_cbow", "ft_sg", "w2v_ft_sg_avg", "glove", "dc_elmo", "dc_bert", "x2_bert", "all"], default="all")
    parser.add_argument("-m", "--model", help="Model to use.", choices=["lstm", "lr", "svm"], default="lstm")
    parser.add_argument("-t", "--train_bs", help="Train batch size, defaults to 32.", default=32, type=int)
    parser.add_argument("-v", "--valid_bs", help="Validation batch size, defaults to 16.", default=16, type=int)
    parser.add_argument("-e", "--epochs", help="Number of epochs, defaults to 5.", default=5, type=int)
    parser.add_argument("-ml", "--max_len", help="Maximum length of a sentence, defaults to 128.", default=128, type=int)
    parser.add_argument("-o", "--output", help="Output file name, defaults to 'output.txt'.", default="output.txt")
    parser.add_argument("-d", "--dataset", help="Dataset #1 corresponds to turkish movie dataset. Dataset #2 corresponds to product reviews.", choices=[1, 2], default = 1, type=int)
    args = parser.parse_args()

    w = args.word_embedding
    train_bs = args.train_bs
    valid_bs = args.valid_bs
    epochs = args.epochs
    max_len = args.max_len
    model = args.model

    traindf, testdf, trainy, testy, negatives, positives = preprocess(args.dataset)
    if w == "all": word_embeddings = ["w2v_sg", "w2v_cbow", "ft_sg", "w2v_ft_sg_avg", "glove", "dc_elmo", "dc_bert"] # "x2_bert"
    else: word_embeddings = [w]

    hist = {}
    for w in word_embeddings:
        config_ = Config(w, max_len, train_bs, valid_bs, epochs)
        tokenizer = tokenize(traindf)
        if(model == "lstm"):
            model_ = train_lstm(tokenizer, config_.word_vectors, config_.dim, traindf, max_len, train_bs, valid_bs, epochs)
            acc, f1 = test_lstm(tokenizer, max_len, testdf, testy, model_)
        elif(model == 'lr'):
            acc, f1 = train_lr(tokenizer, config_.word_vectors, config_.dim, traindf, testdf, max_len)
        elif(model == 'svm'):
            acc, f1 = train_svm(tokenizer, config_.word_vectors, config_.dim, traindf, testdf, max_len)
        print(f"Accuracy for {w} using model {model}: {acc:.3f} with f1 score: {f1:.3f}")
        hist[w] = (acc, f1)

    # write to output
    with open(args.output, "a") as f:
        for w, (acc, f1) in hist.items():
            f.write(f"Accuracy for {w} using model {model}: {acc:.3f} with f1 score: {f1:.3f}\n")

    # IMPORTANT: For LR, average the word embeddings for each review before feeding it to LR