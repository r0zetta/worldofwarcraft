from keras.layers import Input, Dense, Lambda, Layer
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import backend as K
from keras import metrics
from nltk import pos_tag
import nltk.data
from text_handler import load_and_tokenize, split_input_into_sentences
import multiprocessing
import json
import os
import pickle
import itertools
import argparse
import sys
import numpy as np
from scipy import spatial
from scipy.stats import norm
import gensim.models.word2vec as w2v

save_dir = "vae_save"

def save_json(variable, filename):
    with open(filename, "w") as f:
        json.dump(variable, f, indent=4)

def load_json(filename):
    ret = None
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                ret = json.load(f)
            print("Loaded data from " + filename + ".")
        except:
            print("Couldn't load " + filename + ".")
    else:
        print(filename + " didn't exist.")
    return ret

def get_saved_args():
    saved = None
    args_file = os.path.join(save_dir, "args.json")
    if os.path.exists(save_dir):
        if os.path.exists(args_file):
            with open(args_file, "r") as f:
                try:
                    saved = json.load(f)
                except:
                    saved = None
    return saved

def save_args(args):
    args_file = os.path.join(save_dir, "args.json")
    if os.path.exists(save_dir):
        with open(args_file, "w") as f:
            json.dump(args, f, indent=4)

def get_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                       help='train or sample')
    parser.add_argument('--tokenize', type=str, default='chars',
                       help='tokenize by words or chars')
    parser.add_argument('-n', type=int, default=1,
                       help='The number of sentences to generate')
    parser.add_argument('--batch_size', type=int, default=300,
                       help='minibatch size')
    parser.add_argument('--latent_dim', type=int, default=1000,
                       help='latent dimension')
    parser.add_argument('--intermediate_dim', type=int, default=1200,
                       help='intermediate dimension')
    parser.add_argument('--num_epochs', type=int, default=500,
                       help='number of epochs')
    parser.add_argument('--num_features', type=int, default=10,
                       help='number of features in w2v model')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='learning rate')
    parser.add_argument('--epsilon_std', type=float, default=1.0,
                       help='epsilon std')
    args = parser.parse_args()
    return vars(args)

def get_args():
    args = None
    saved_args = get_saved_args()
    cl_args = get_cl_args()
    if saved_args is not None:
        args = saved_args
    else:
        args = cl_args
    args["mode"] = cl_args["mode"]
    args["n"] = cl_args["n"]
    print("Args:")
    for key, value in sorted(args.iteritems()):
        print("\t" + str(key) + ":\t" + str(value))
    save_args(args)
    return args

def create_vae_model(args):
    batch_size = args["batch_size"]
    latent_dim = args["latent_dim"]
    intermediate_dim = args["intermediate_dim"]
    epsilon_std = args["epsilon_std"]
    input_size = args["input_size"]
    original_dim = args["original_dim"]

    x = Input(batch_shape=(input_size, original_dim))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(input_size, latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    def zero_loss(y_true, y_pred):
        return K.zeros_like(y_pred)

    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            return K.ones_like(x)

    loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, [loss_layer])
    vae.compile(optimizer='rmsprop', loss=[zero_loss])

# build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

# build a generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    return vae, encoder, generator

def create_w2v_model(args):
    num_features = args["num_features"]
    input_file = args["raw_data_file"]
    min_word_count = 0
    num_workers = multiprocessing.cpu_count()
    context_size = 7
    downsampling = 0
    seed = 1
    epoch_count = 10

    print("Training word2vec model")
    print("Features per word: " + str(num_features))
    print("Loading raw data")
    raw_data = load_json(input_file)
    print("Tokenizing data")
    sentences = split_input_into_sentences(raw_data)

    sentence_count = len(sentences)
    print("Number of sentences: " + str(sentence_count))
    token_count = sum([len(sentence) for sentence in sentences])
    print("The corpus contains " + str(token_count) + " tokens")

    word_vectors = w2v.Word2Vec(sg=1,
                                seed=seed,
                                workers=num_workers,
                                size=num_features,
                                min_count=min_word_count,
                                window=context_size,
                                sample=downsampling)

    print("Building vocab...")
    word_vectors.build_vocab(sentences)

    print("Word2Vec vocabulary length:", len(word_vectors.wv.vocab))

    print("Training...")
    word_vectors.train(sentences, total_examples=sentence_count, epochs=epoch_count)

    model_filename = os.path.join(save_dir, "word_vectors.w2v")
    print("Saving model...")
    word_vectors.save(model_filename)


def prepare_data(args):
    print("Loading word2vec model")
    w2v_file = os.path.join(save_dir, "word_vectors.w2v")
    word2vec = None
    if not os.path.exists(w2v_file):
        create_w2v_model(args)

    word2vec = w2v.Word2Vec.load(w2v_file)

    all_word_vectors_matrix = word2vec.wv.syn0
    num_words = len(all_word_vectors_matrix)
    print("Number of word vectors: " + str(num_words))

    vocab = word2vec.wv.vocab
    print("Vocab length: " + str(len(vocab)))
    dim = word2vec.wv["."].shape
    print("Dim: " + str(dim))
    dim0 = word2vec.wv["."].shape[0]
    print("Features per word: " + str(dim0))

    print("Loading training data")
    tokens_file = os.path.join(save_dir, "tokens.json")
    tokenized_data = load_json(tokens_file)
    if tokenized_data is None:
        tokenized_data = load_and_tokenize(args["raw_data_file"], "words")
        if tokenized_data is None:
            assert False, "Could not process raw data into tokens."
        save_json(tokenized_data, tokens_file)
    else:
        print("Loaded tokenized data from file.")

    print("Vectorizing data")
    vectors = []
    omitted = []
    not_added = 0
    added = 0
    for token in tokenized_data:
        try:
            vect = word2vec.wv[token]
            vectors.append(vect)
            added += 1
        except:
            if len(token) > 0:
                omitted.append(token)
            not_added += 1
            pass

    print("Added: " + str(added) + " words to vector.")
    print("Failed to add: " + str(not_added) + " words to vector.")
    print("Omitted:")
    print omitted

    num_vectors = len(vectors)
    print("num_vectors: " + str(num_vectors))
    print("Creating batches")
    batch_pointer = 0
    batch_size = int(args["batch_size"])
    batch_len = batch_size * dim0
    num_batches = num_vectors/batch_size
    batches = []
    print("Number of batches: " + str(num_batches))
    for x in range(num_batches):
        batch = []
        subset = vectors[batch_pointer: batch_pointer + batch_size]
        if len(subset) == 0:
            print("Vector subset was zero")
            sys.exit(0)
        for item in subset:
            for vec in item:
                batch.append(vec)
        if len(batch) == 0:
            print("Batch was empty")
            sys.exit(0)
        batches.append(batch)
        batch_pointer += batch_size
    data_array = np.array(batches)
    print("Full data shape: " + str(data_array.shape))
    np.random.shuffle(data_array)
    data_len = len(data_array)
    train_len = int(data_len * 0.8)

    train = data_array[:train_len]
    print("Training data shape: " + str(train.shape))
    test = data_array[train_len:]
    print("Test data shape: " + str(test.shape))

    return train, test, word2vec, dim0


def decode_sentence(vectors, args, word2vec):
    vector_count = len(vectors)
    print("Vectors: " + str(vector_count))
    features = args["num_features"]
    print("Features: " + str(features))
    num_words = vector_count/features
    print("Decoding " + str(num_words) + " words")
    decoded_words = []
    for x in range(num_words):
        start_pos = x * features
        word_as_vectors = vectors[start_pos: start_pos + features]
        model_word_vector = np.array(word_as_vectors, dtype='f')
        #print model_word_vector
        if len(model_word_vector) == 0:
            print("Vector was empty. WTF???!??!?")
        topn = 1;
        word = word2vec.most_similar( [ model_word_vector ], [], topn)
        decoded_words.append(word[0][0])
    return decoded_words

# input: encoded sentence vector
# output: encoded sentence vector in dataset with highest cosine similarity
def find_similar_encoding(sent_vect):
    all_cosine = []
    for sent in sent_encoded:
        result = 1 - spatial.distance.cosine(sent_vect, sent)
        all_cosine.append(result)
    data_array = np.array(all_cosine)
    maximum = data_array.argsort()[-3:][::-1][1]
    new_vec = sent_encoded[maximum]
    return new_vec

# input: two points, integer n
# output: n equidistant points on the line between the input points (inclusive)
def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample

#########################################################
# Entry point
#########################################################
if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args = get_args()
    args["raw_data_file"] = os.path.join(save_dir, "data.json")

    train, test, word2vec, features = prepare_data(args)
    args["num_features"] = features
    batch_len = features * args["batch_size"]
    print("".join(decode_sentence(train[0], args, word2vec)))

    args["input_size"] = input_size = 1
    args["original_dim"] = original_dim = len(train[0])
    epochs = args["num_epochs"]

    vae, encoder, generator = create_vae_model(args)
    checkpoint_file = os.path.join(save_dir, "model.h5")
    if os.path.exists(checkpoint_file):
        print("Loading saved model from " + checkpoint_file)
        vae.load_weights(checkpoint_file)
    cp = [ModelCheckpoint(filepath=checkpoint_file, verbose=1, save_best_only=True)]

    if args["mode"] == "train":
        vae.fit(train, train,
                shuffle=True,
                epochs=epochs,
                batch_size=input_size,
                validation_data=(test, test), callbacks=cp)
    else:
        print("Running predict model")
        sent_encoded = encoder.predict(np.array(train), batch_size = 1)
        print("".join(decode_sentence(sent_encoded[0], args, word2vec)))
        print("Running generator model")
        sent_decoded = generator.predict(sent_encoded)
        print("".join(decode_sentence(sent_decoded[0], args, word2vec)))

# The encoder trained above embeds sentences (concatenated word vetors) into a lower dimensional space. The code below takes two of these lower dimensional sentence representations and finds five points between them. It then uses the trained decoder to project these five points into the higher, original, dimensional space. Finally, it reveals the text represented by the five generated sentence vectors by taking each word vector concatenated inside and finding the text associated with it in the word2vec used during preprocessing.

        print("Running shortest homology test 1")
        test_hom = shortest_homology(sent_encoded[3], sent_encoded[10], 5)
        for point in test_hom:
            p = generator.predict(np.array([point]))[0]
            print("".join(decode_sentence(p, args, word2vec)))


# The code below does the same thing, with one important difference. After sampling equidistant points in the latent space between two sentence embeddings, it finds the embeddings from our encoded dataset those points are most similar to. It then prints the text associated with those vectors.
#   
# This allows us to explore how the Variational Autoencoder clusters our dataset of sentences in latent space. It lets us investigate whether sentences with similar concepts or grammatical styles are represented in similar areas of the lower dimensional space.

        print("Running shortest homology test 2")
        test_hom = shortest_homology(sent_encoded[2], sent_encoded[1500], 20)
        for point in test_hom:
            p = generator.predict(np.array([find_similar_encoding(point)]))[0]
            print("".join(decode_sentence(p, args, word2vec)))

