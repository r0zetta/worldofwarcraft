from keras.layers import Input, Dense, Lambda, Layer
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import backend as K
from keras import metrics
from nltk import pos_tag
import nltk.data
from text_handler import split_input_into_sentences
import multiprocessing
import json
import time
import os
import pickle
import itertools
import argparse
import sys
import random
import numpy as np
from scipy import spatial
from scipy.stats import norm
import gensim.models.word2vec as w2v

save_dir = "vae_save"
char_input_size = 1000
word_input_size = 100

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
    parser.add_argument('--tokenize', type=str, default='words',
                       help='tokenize by words or chars')
    parser.add_argument('--intermediate_activator', type=str, default='relu',
                       help='activation type (relu, sigmoid, tanh)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       help='optimizer (rmsprop, adam, etc.)')
    parser.add_argument('-n', type=int, default=1,
                       help='The number of sentences to generate')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                       help='learning rate')
    parser.add_argument('--verbosity', type=int, default=1,
                       help='keras fit function verbosity')
    parser.add_argument('--sample_every', type=int, default=20,
                       help='run sample every x epochs')
    parser.add_argument('--num_epochs', type=int, default=3000,
                       help='number of epochs')
    parser.add_argument('--num_features', type=int, default=50,
                       help='number of features in w2v model')
    parser.add_argument('--magic', type=float, default=0.003,
                       help='magic number')
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

def create_w2v_model(args, sentences):
    w2v_file = os.path.join(save_dir, "word_vectors.w2v")
    if os.path.exists(w2v_file):
        print("w2v model loaded from " + w2v_file)
        word2vec = w2v.Word2Vec.load(w2v_file)
        return word2vec

    print("Building word2vec model")
    num_features = args["num_features"]
    print("Features per word: " + str(num_features))
    num_workers = multiprocessing.cpu_count()
    epoch_count = 10

    sentence_count = len(sentences)
    print("Number of sentences: " + str(sentence_count))
    token_count = sum([len(sentence) for sentence in sentences])
    print("The corpus contains " + str(token_count) + " tokens")

    word_vectors = w2v.Word2Vec(sg=1,
                                seed=1,
                                workers=num_workers,
                                size=num_features,
                                min_count=0,
                                window=7,
                                sample=0)

    print("Building vocab...")
    word_vectors.build_vocab(sentences)
    print("Word2Vec vocabulary length:", len(word_vectors.wv.vocab))
    print("Training...")
    word_vectors.train(sentences, total_examples=sentence_count, epochs=epoch_count)

    print("Saving model...")
    model_filename = os.path.join(save_dir, "word_vectors.w2v")
    word_vectors.save(model_filename)
    return word_vectors

def split_words_into_chars(words):
    ret = []
    for word in words:
        print word
        entry = []
        for c in list(word):
            entry.append(c)
        ret.append(entry)
    return ret

def split_and_pad_input(input_data, args):
    items = []
    input_size = 0
    if args["tokenize"] == "words":
        items = split_input_into_sentences(input_data)
        input_size = word_input_size
    else:
        items = split_words_into_chars(input_data)
        input_size = char_input_size
    max_elements = len(items)
    while max_elements % input_size != 0:
        max_elements -= 1
    print("Using " + str(max_elements) + " elements")
    random.shuffle(items)
    items = items[:max_elements]
    max_item_len = len(max(items, key=len))
    print("Longest item: " + str(max_item_len))
    ret = []
    for item in items:
        entry = []
        item_len = len(item)
        padding_len = max_item_len - item_len
        for element in item:
            entry.append(element)
        for _ in range(padding_len):
            entry.append("")
        ret.append(entry)
    return ret

def vectorize_data(args, tokenized_data, word2vec):
    print("Vectorizing data")

    dim0 = word2vec.wv[""].shape[0]
    null_entry = np.zeros([dim0], dtype=np.float32)
    print("Number of features: " + str(dim0))

    max_name_len = len(max(tokenized_data, key=len))
    print("Maximum name length: " + str(max_name_len))

    batch_len = max_name_len * dim0
    print("Batch length: " + str(batch_len))

    print("Vectorizing and padding data")
    vectorized = []
    not_added = []
    for tokens in tokenized_data:
        entry = []
        if len(tokens) != max_name_len:
            print len(tokens)
            print tokens
        assert len(tokens) == max_name_len
        for t in tokens:
            try:
                vect = word2vec.wv[t]
                entry.append(vect)
            except:
                not_added.append(t)
                pass
        if len(entry) != max_name_len:
            print entry
            print len(entry)
        assert len(entry) == max_name_len
        vectorized.append(entry)

    print(str(len(not_added)) + " items omitted during vectorization.")
    print not_added

    print("Creating batches")
    batches = []
    for vectors in vectorized:
        entry = []
        for vec_list in vectors:
            for vec in vec_list:
                entry.append(vec)
        if len(entry) != batch_len:
            print entry
            print len(entry)
        assert len(entry) == batch_len
        batches.append(entry)

    print(str(len(batches)) + " batches created.")
    data_array = np.array(batches)
    print("Full data shape: " + str(data_array.shape))
    return data_array

def prepare_data(args):
    print("Loading training data")
    tokens_file = os.path.join(save_dir, "tokens.json")
    tokenized_data = load_json(tokens_file)
    if tokenized_data is None:
        raw_data = load_json(args["raw_data_file"])
        if raw_data is None:
            assert False, "Could not load raw data."
        tokenized_data = split_and_pad_input(raw_data, args)
        if tokenized_data is None:
            assert False, "Could not process raw data into tokens."
        save_json(tokenized_data, tokens_file)
    else:
        print("Loaded tokenized data from file.")

    word2vec = create_w2v_model(args, tokenized_data)
    all_word_vectors_matrix = word2vec.wv.syn0
    num_words = len(all_word_vectors_matrix)
    print("Number of word vectors: " + str(num_words))

    vocab = word2vec.wv.vocab
    print("Vocab length: " + str(len(vocab)))
    dim = word2vec.wv[""].shape
    print("Dim: " + str(dim))
    dim0 = word2vec.wv[""].shape[0]
    print("Features per word: " + str(dim0))

    data_array = vectorize_data(args, tokenized_data, word2vec)
    np.random.shuffle(data_array)
    data_len = len(data_array)

    return data_array, word2vec, dim0, tokenized_data

def encode_sentence(sentence, args, word2vec):
    tokenized = []
    for c in list(sentence):
        try:
            vecs = word2vec.wv[c]
            for v in vecs:
                tokenized.append(v)
        except:
            pass
    return tokenized

def decode_sentence(vectors, args, word2vec):
    vector_count = len(vectors)
    #print("Vectors: " + str(vector_count))
    features = args["num_features"]
    #print("Features: " + str(features))
    num_words = vector_count/features
    #print("Decoding " + str(num_words) + " items")
    decoded_words = []
    for x in range(num_words):
        start_pos = x * features
        word_as_vectors = vectors[start_pos: start_pos + features]
        model_word_vector = np.array(word_as_vectors, dtype='f')
        #print("Word:" + str(x) + " shape:" + str(model_word_vector.shape))
        #print model_word_vector
        if len(model_word_vector) == 0:
            print("Vector was empty. WTF???!??!?")
        topn = 1;
        word = word2vec.most_similar( [ model_word_vector ], [], topn)
        decoded_words.append(word[0][0])
    return decoded_words

def test_decoder(args, word2vec):
    raw_data = load_json(args["raw_data_file"])
    orig_data = raw_data
    orig_data_len = len(orig_data)
    for _ in range(10):
        index = random.randint(0, orig_data_len-1)
        orig_item = orig_data[index]
        encoded = encode_sentence(orig_item, args, word2vec)
        decoded = decode_sentence(encoded, args, word2vec)
        decoded_item = "".join(decoded)
        print(orig_item + " = " + decoded_item)

def sample(train, args, word2vec, encoder, generator):
    sep = " "
    decode_sep = "\n"
    raw_data = load_json(args["raw_data_file"])
    original_data = raw_data[0].split(" ")
    if args["tokenize"] == "chars":
        sep = ""
        decode_sep = " "
    sent_encoded = encoder.predict(np.array(train), batch_size = args["input_size"])
    #print("Obtained " + str(len(sent_encoded)) + " encoded items from predict.")
    sent_decoded = generator.predict(sent_encoded)
    print("Obtained " + str(len(sent_decoded)) + " decoded items from generator.")
    #print("Decoded length: " + str(len(sent_decoded[0])))
    output = []
    for x in range(len(sent_decoded)):
        output.append(sep.join(decode_sentence(sent_decoded[x], args, word2vec)))
    outstr = ""
    for _ in range(100):
        index = random.randint(0, len(sent_decoded)-1)
        outstr += output[index] + " "
    print outstr
    timestamp = int(time.time())
    filename = os.path.join(save_dir, "output_" + str(timestamp) + ".json")
    save_json(output, filename)

def choose_tokenize_mode(args):
    raw_data = load_json(args["raw_data_file"])
    max_len = len(max(raw_data, key=len))
    if max_len > 50:
        return "words"
    else:
        return "chars"

#########################################################
# Entry point
#########################################################
if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args = get_args()
# raw data should be a json file that contains a list like ["name", "name", "name"...]
# or a list like ["sentence", "sentence", "sentence"...]
    args["raw_data_file"] = os.path.join("data", "data.json")
    train_state_file = os.path.join(save_dir, "train_state.json")
    current_epoch = 0
    if os.path.exists(train_state_file):
        current_epoch = load_json(train_state_file)

    random.seed(1)
    args["tokenize"] = choose_tokenize_mode(args)
    print("Tokenize mode: " + args["tokenize"])
    train, word2vec, features, tokens = prepare_data(args)
    args["num_features"] = features
    print
    print("Testing decoder")
    test_decoder(args, word2vec)
    print

    args["original_dim"] = original_dim = len(train[0])
    epochs = args["num_epochs"]
    epochs_per_iter = args["sample_every"]
    input_size = word_input_size
    if args["tokenize"] == "chars":
        input_size = char_input_size

    args["input_size"] = input_size
    intermediate_dim = int(original_dim / 3)
    latent_dim = int(intermediate_dim * 1.2)
    epsilon_std = args["epsilon_std"]
    magic = args["magic"]
    intermediate_activator = args["intermediate_activator"]
    optimizer = args["optimizer"]
    learning_rate = args["learning_rate"]
    verbosity = args["verbosity"]
    save_args(args)

    x = Input(batch_shape=(input_size, original_dim))
    h = Dense(intermediate_dim, activation=intermediate_activator)(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    print("Parameters")
    print("==========")
    print("Batch count: " + str(input_size))
    print("Input dim: " + str(original_dim))
    print("Intermediate dim: " + str(intermediate_dim))
    print("Latent dim: " + str(latent_dim))
    print("epsilon_std: " + str(epsilon_std))
    print("magic: " + str(magic))
    print("learning rate: " + str(learning_rate))
    print("intermediate_activator: " + str(intermediate_activator))
    print("optimizer: " + str(optimizer))
    print("epochs: " + str(epochs))
    print("epochs_per_iter: " + str(epochs_per_iter))
    print

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(input_size, latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoder_h = Dense(intermediate_dim, activation=intermediate_activator)
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
            kl_loss = - magic * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            return K.ones_like(x)

    loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, [loss_layer])
    print("Compiling model")
    vae.compile(optimizer=optimizer, loss=[zero_loss])

# build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

# build a generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    checkpoint_file = os.path.join(save_dir, "model.h5")
    if os.path.exists(checkpoint_file):
        print("Loading saved model from " + checkpoint_file)
        vae.load_weights(checkpoint_file)
    cp = [ModelCheckpoint(filepath=checkpoint_file, verbose=0, save_best_only=True)]

    if args["mode"] == "train":
        print("Engaging training mode.")
        while current_epoch < epochs:
            print("Current epoch: " + str(current_epoch) + "/" + str(epochs))
            vae.fit(train, train,
                    shuffle=True,
                    epochs=epochs_per_iter,
                    batch_size=input_size,
                    verbose=verbosity,
                    validation_data=(train, train),
                    callbacks=cp)
            current_epoch += epochs_per_iter
            save_json(current_epoch, train_state_file)
            sample(train, args, word2vec, encoder, generator)
    else:
        sample(train, args, word2vec, encoder, generator)


