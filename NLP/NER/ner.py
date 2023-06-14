import numpy as np
import pandas as pd
import tensorflow as tf
import gensim
import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import Dataset
from sklearn import metrics
from torchmetrics.classification import F1Score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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

def read_only_consistent(path):
    # create a dataframe consisting of word and PoS tag
    sentences = []
    with open(path, 'r', encoding="utf-8") as f:
        data = f.readlines()
        r = len(data)
        i = 0
        while(i < r):
            sentence = []
            ne_representation = []
            while(i < r and data[i] != "\n"):
                parts = data[i].split(" ")
                word = parts[0].strip()
                ne = parts[-1].strip()

                ne_representation.append(ne)
                sentence.append(word) 
                i += 1
            i+=1
            sentences.append((sentence, ne_representation))
        return sentences

def preprocess(dataset):
    # read test, dev and train csv files
    train = read_only_consistent("data/gungor.ner.train.14.only_consistent")
    dev = read_only_consistent("data/gungor.ner.dev.14.only_consistent")
    test = read_only_consistent("data/gungor.ner.test.14.only_consistent")

        # create a dataframe
    traindf = pd.DataFrame(
        {'sentence': [i[0] for i in train],
            'ne': [i[1] for i in train]
        })

    testdf = pd.DataFrame(
        {'sentence': [i[0] for i in test],
            'ne': [i[1] for i in test]
        })

    devdf = pd.DataFrame(
        {'sentence': [i[0] for i in dev],
            'ne': [i[1] for i in dev]
        })
    
    return traindf, testdf, devdf

def vectorize(traindf):
    trainx = traindf['sentence']
    trainy = traindf['ne']
    ####### VECTORIZATION #######
    word_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    word_tokenizer.fit_on_texts(trainx)                    # fit tokeniser on data
    X_encoded = word_tokenizer.texts_to_sequences(trainx)  # use the tokeniser to encode input sequence

    ####### VECTORIZATION #######
    tag_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tag_tokenizer.fit_on_texts(trainy)                    # fit tokeniser on data
    Y_encoded = tag_tokenizer.texts_to_sequences(trainy)  # use the tokeniser to encode input sequence

    return word_tokenizer, X_encoded, tag_tokenizer, Y_encoded

def pad(max_len, X_encoded, Y_encoded):
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(X_encoded, maxlen=max_len, padding="pre", truncating="post")
    Y_padded = tf.keras.preprocessing.sequence.pad_sequences(Y_encoded, maxlen=max_len, padding="pre", truncating="post")

    Y_final = tf.keras.utils.to_categorical(Y_padded)

    return X_padded, Y_final, Y_padded

def create_embeddings(vocab_size, embed_size, wv):
    embedding_weights = np.zeros((vocab_size, embed_size))
    word2id = word_tokenizer.word_index
    for word, index in word2id.items():
        try:  embedding_weights[index, :] = wv[word]
        except KeyError: pass
    
    return embedding_weights

def pipeline(df, word_tokenizer, tag_tokenizer, maxlen):
    dfx = df['sentence']
    dfy = df['ne']

    dfx_encoded = word_tokenizer.texts_to_sequences(dfx)
    dfy_encoded = tag_tokenizer.texts_to_sequences(dfy)

    dfx_padded = tf.keras.preprocessing.sequence.pad_sequences(dfx_encoded, maxlen=maxlen, padding="pre", truncating="post")
    dfy_padded = tf.keras.preprocessing.sequence.pad_sequences(dfy_encoded, maxlen=maxlen, padding="pre", truncating="post")

    dfy_final= tf.keras.utils.to_categorical(dfy_padded)

    return dfx_padded, dfy_final, dfy_padded

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
        sentence = data['sentence']
        tags = data['tags']
        # move the data to device that we want to use
        sentence = sentence.to(device, dtype = torch.long)
        tags = tags.to(device, dtype = torch.float)

        
        # clear the gradient
        optimizer.zero_grad()
        # make prediction from model
        
        # sentence shape = batch size, maximum length
        # tag shape = batch size, maximum length, number of tags (one-hot encoded)
        predictions = model(sentence)
        # caculate the losses
        loss = nn.CrossEntropyLoss()(predictions, tags)
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
            sentence = data['sentence']
            targets = data['tags']
            sentence = sentence.to(device, dtype = torch.long)
            
            # make prediction
            predictions = model(sentence)
            # move prediction and target to cpu
            predictions = predictions.cpu().numpy().tolist()
            targets = targets.cpu().numpy().tolist()
            # add predictions to final_prediction
            final_predictions.extend(predictions)
            final_targets.extend(targets)

    return torch.tensor(final_predictions), torch.tensor(final_targets)

class LSTM(nn.Module):
    def __init__(self, embedding_matrix, num_classes):
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
        # Because we use pretrained embedding (GLove, Fastext, etc) so we turn off requires_grad-meaning we do not train gradient on embedding weight
        self.embedding.weight.requires_grad = False
        # LSTM with hidden_size = 16
        self.lstm = nn.LSTM(
                            embedding_dim, 
                            16,
                            bidirectional=False,
                            batch_first=True,
                             )
        self.out = nn.Linear(16, num_classes)
    def forward(self, x):
        # pass input (tokens) through embedding layer

        # x: batch size, max len
        x = self.embedding(x)
        # x: batch size, max len, embedding dimensions
        # fit embedding to LSTM
        hidden, _ = self.lstm(x)
        # x: batch size, max len, # of hidden unit
        # fit out to self.out to conduct dimensionality reduction from 512 to 1
        out = self.out(hidden)
        # return output
        return out

def train_lstm(X_padded, Y_final, devx_padded, devy_final, train_bs, valid_bs, epochs, word_tokenizer):
    vocab_size = len(word_tokenizer.word_index) + 1
    embedding_matrix = create_embeddings(vocab_size, config_.dim, config_.word_vectors)

    # STEP 4: initialize dataset class for training
    train_dataset = Dataset(X=X_padded, y=Y_final)
    
    # STEP 5: Load dataset to Pytorch DataLoader
    # after we have train_dataset, we create a torch dataloader to load train_dataset class based on specified batch_size
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = train_bs, num_workers=2)
    valid_dataset = Dataset(X=devx_padded, y=devy_final)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = valid_bs, num_workers=1)
    
    # STEP 6: Running 
    device = torch.device('cuda')
    model = LSTM(embedding_matrix, num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print('training model')
    for epoch in tqdm(range(epochs)):
        training(train_data_loader, model, optimizer, device)
        outputs, targets = evaluate(valid_data_loader, model, device)
        acc = (torch.argmax(outputs, -1) == torch.argmax(targets, -1)).float().mean()

        print(f'epoch: {epoch}, Validation Accuracy Score: {acc}')
    return model


def test_lstm(testx_padded, testy_final, model):
    device = torch.device('cuda')

    test_dataset = Dataset(X=testx_padded, y=testy_final)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = valid_bs, num_workers=1)
    
    outputs, targets = evaluate(test_data_loader, model, device)
    acc = (torch.argmax(outputs, -1) == torch.argmax(targets, -1)).float().mean()
    return acc 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--word_embedding", help="Word embedding file to experiment on.", choices=["w2v_sg", "w2v_cbow", "ft_sg", "w2v_ft_sg_avg", "glove", "dc_elmo", "dc_bert", "x2_bert", "all"], default="all")
    parser.add_argument("-t", "--train_bs", help="Train batch size, defaults to 32.", default=32, type=int)
    parser.add_argument("-v", "--valid_bs", help="Validation batch size, defaults to 16.", default=16, type=int)
    parser.add_argument("-e", "--epochs", help="Number of epochs, defaults to 5.", default=5, type=int)
    parser.add_argument("-ml", "--max_len", help="Maximum length of a sentence, defaults to 128.", default=128, type=int)
    parser.add_argument("-o", "--output", help="Output file name, defaults to 'output.txt'.", default="output.txt")
    parser.add_argument("-d", "--dataset", help="Dataset #1 is prepared by Gungor. Dataset #2 corresponds to product reviews.", choices=[1, 2], default = 1)
    args = parser.parse_args()

    w = args.word_embedding
    train_bs = args.train_bs
    valid_bs = args.valid_bs
    epochs = args.epochs
    max_len = args.max_len

    traindf, testdf, devdf = preprocess(args.dataset)
    if w == "all": word_embeddings = ["w2v_sg", "w2v_cbow", "ft_sg", "w2v_ft_sg_avg", "glove", "dc_elmo", "dc_bert"] # "x2_bert"
    else: word_embeddings = [w]

    word_tokenizer, X_encoded, tag_tokenizer, Y_encoded = vectorize(traindf)
    X_padded, Y_final, Y_padded = pad(max_len, X_encoded, Y_encoded)
    
    devx_padded, devy_final, _ = pipeline(devdf, word_tokenizer, tag_tokenizer, max_len)
    testx_padded, testy_final, testy_padded = pipeline(testdf, word_tokenizer, tag_tokenizer, max_len)
    num_classes = Y_final.shape[2]

    hist = {}
    for w in word_embeddings:
        config_ = Config(w, max_len, train_bs, valid_bs, epochs)

        model_ = train_lstm(X_padded, Y_final, devx_padded, devy_final, train_bs, valid_bs, epochs, word_tokenizer)
        acc = test_lstm(testx_padded, testy_final, model_)        
        print(f"Accuracy for {w} using lstm: {acc:.3f}")
        hist[w] = acc

    # write to output
    with open(args.output, "a") as f:
        for w, acc in hist.items():
            f.write(f"Accuracy for {w} using lstm: {acc:.3f}\n")
