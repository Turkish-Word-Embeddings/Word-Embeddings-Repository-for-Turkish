# Word2Vec Negative Sample implementation
# paper: https://arxiv.org/abs/1301.3781
# source: https://www.tensorflow.org/tutorials/text/word2vec

import tensorflow as tf
import io
import re 
import string
import numpy as np
import argparse

# Model
class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, num_ns=4):
    super(Word2Vec, self).__init__()
    self.target_embedding = tf.keras.layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = tf.keras.layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=num_ns+1)

  def call(self, pair):
    target, context = pair

    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)               # word_emb: (batch size, embedding size)
    context_emb = self.context_embedding(context)          # context_emb: (batch size, context size, embedding size)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)  # dots: (batch size, context size)
    return dots

def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples and vocabulary size.
def generate_training_data(sequences, window_size, num_negative_samples, vocab_size):
  targets, contexts, labels = [], [], []
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)  #  Zipf's distribution

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in sequences:
    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_negative_samples,
          unique=True,
          range_max=vocab_size,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_negative_samples, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,'[%s]' % re.escape(string.punctuation), '')

# simple usage: python word2vec/tf_w2v.py -c "corpus/Turkish-English Parallel Corpus.txt"  
if __name__ == '__main__':
    # take input corpora file, batch size, embedding dimensions and output file name as command line arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpora", help="corpora file", required=True)
    parser.add_argument("-b", "--batch_size",  help="batch size, defaults to 1024 if not provided", default=1024, type=int)
    parser.add_argument("-d", "--embedding_dimensions",  help="embedding dimensions, defaults to 128", default=128, type=int)
    parser.add_argument("-o", "--output", help="output folder to store tsv files, defaults to current directory if not provided", default='.')
    parser.add_argument("-e", "--epoch",  help="number of epochs, defaults to 20 if not provided", default=20, type=int)
    parser.add_argument("-m", "--maximum_sentence_length",  help="maximum sentence (sequence) length, defaults to 300 if not provided", type=int, default=300)
    parser.add_argument("-v", "--vocab_size",  help="vocabulary size, defaults to 300,000 if not provided", default=300000, type=int)
    parser.add_argument("-n", "--negative_sampling",  help="number of negative samples, defaults to 4 (assuming large datasets)", type=int, default=4)
    parser.add_argument("-w", "--window_size",  help="window size, defaults to 2", default=2, type=int)
    args = parser.parse_args()
    
    corpora_file = args.corpora
    batch_size = args.batch_size
    embedding_dimensions = args.embedding_dimensions
    output = args.output
    epochs = args.epoch
    maximum_sentence_length = args.maximum_sentence_length
    vocab_size = args.vocab_size
    negative_sampling = args.negative_sampling
    window_size = args.window_size
    # read corporus into a dataset
    dataset = tf.data.TextLineDataset(corpora_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))
    # find maximum length of a sentence in the dataset
    vectorize_layer = tf.keras.layers.TextVectorization(
                    standardize=custom_standardization,
                    max_tokens=vocab_size,
                    output_mode='int',
                    output_sequence_length=maximum_sentence_length)
                    
    vectorize_layer.adapt(dataset.batch(batch_size))  # ~1.5 minutes
    # Vectorize the data in dataset.
    text_vector_ds = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()
    sequences = list(text_vector_ds.as_numpy_iterator())

    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=window_size,
        num_negative_samples=negative_sampling,
        vocab_size=vocab_size)

    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)

    buffer_size = 10000
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    word2vec = Word2Vec(vocab_size, embedding_dimensions)
    word2vec.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    
    callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit(dataset, epochs=epochs, callbacks=[callback], verbose=1)

    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    out_v = io.open('{}/vectors.tsv'.format(output), 'w', encoding='utf-8')
    out_m = io.open('{}/metadata.tsv'.format(output), 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if (index == 0): continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()