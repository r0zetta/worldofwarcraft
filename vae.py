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
    parser.add_argument('--intermediate_activator', type=str, default='relu',
                       help='activation type (relu, sigmoid, tanh)')
    parser.add_argument('--optimizer', type=str, default='rmsprop',
                       help='optimizer (rmsprop, adam, etc.)')
    parser.add_argument('-n', type=int, default=1,
                       help='The number of sentences to generate')
    parser.add_argument('--batch_size', type=int, default=300,
                       help='minibatch size')
    parser.add_argument('--sample_every', type=int, default=200,
                       help='run sample every x epochs')
    parser.add_argument('--num_epochs', type=int, default=300,
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


def split_names_into_chars(input_data):
    words = input_data[0].split(" ")
    print("Got " + str(len(words)) + " words")
    max_words = len(words)
    while max_words % 1000 != 0:
        max_words -= 1
    print("Using " + str(max_words) + " words")
    max_word_len = 0
    for w in words:
        if len(w) > max_word_len:
            max_word_len = len(w)
    sentences = []
    word_count = 1
    vocab = []
    for w in words:
        sent = []
        word_len = len(w)
        padding_len = max_word_len - word_len
        for c in list(w):
            sent.append(c)
            if c not in vocab:
                vocab.append(c)
        for _ in range(padding_len):
            sent.append("")
        sentences.append(sent)
        if word_count >= max_words:
            break
        word_count += 1
    vocab_len = len(vocab) + 1
    print("Vocab len: " + str(vocab_len))
    print("Got " + str(len(sentences)) + " sentences")
    return sentences, vocab_len

def create_w2v_model(args):
    w2v_file = os.path.join(save_dir, "word_vectors.w2v")
    if os.path.exists(w2v_file):
        print("w2v model loaded from " + w2v_file)
        word2vec = w2v.Word2Vec.load(w2v_file)
        return word2vec

    print("Building word2vec model")
    num_features = args["num_features"]
    input_file = args["raw_data_file"]
    num_workers = multiprocessing.cpu_count()
    epoch_count = 10

    print("Features per word: " + str(num_features))
    print("Loading raw data")
    raw_data = load_json(input_file)
    print("Tokenizing data")
    sentences = []
    if args["tokenize"] == "words":
        sentences = split_input_into_sentences(raw_data)
    else:
        sentences, vocab_len = split_names_into_chars(raw_data)

    print("Number of features: " + str(num_features))
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

def vectorize_names(args, tokenized_data, word2vec):
    print("Vectorizing names")
    dim0 = word2vec.wv["a"].shape[0]
    max_name_len = 0
    null_entry = np.zeros([dim0], dtype=np.float32)
    for t in tokenized_data:
        if len(t) > max_name_len:
            max_name_len = len(t)
    batch_len = max_name_len * dim0

    print("Maximum name length: " + str(max_name_len))
    print("Number of features: " + str(max_name_len))
    print("Batch length: " + str(batch_len))

    print("Vectorizing and padding data")
    vec_names = []
    not_added = []
    for tokens in tokenized_data:
        entry = []
        if len(tokens) != max_name_len:
            print tokens
            print len(tokens)
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
        vec_names.append(entry)

    print(str(len(not_added)) + " characters omitted")
    print not_added

    print("Creating batches")
    batches = []
    for name in vec_names:
        entry = []
        for vec_list in name:
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

def vectorize_sentences(args, tokenized_data, word2vec):
    print("Vectorizing sentences")
    vectors = []
    omitted = []
    not_added = 0
    added = 0
    dim0 = word2vec.wv["a"].shape[0]
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
    return data_array

def prepare_data(args):
    word2vec = create_w2v_model(args)
    all_word_vectors_matrix = word2vec.wv.syn0
    num_words = len(all_word_vectors_matrix)
    print("Number of word vectors: " + str(num_words))

    vocab = word2vec.wv.vocab
    print("Vocab length: " + str(len(vocab)))
    dim = word2vec.wv["a"].shape
    print("Dim: " + str(dim))
    dim0 = word2vec.wv["a"].shape[0]
    print("Features per word: " + str(dim0))

    print("Loading training data")
    tokens_file = os.path.join(save_dir, "tokens.json")
    tokenized_data = load_json(tokens_file)
    if tokenized_data is None:
        if args["tokenize"] == "words":
            tokenized_data = load_and_tokenize(args["raw_data_file"], "words")
        else:
            raw_data = load_json(args["raw_data_file"])
            tokenized_data, vocab_len = split_names_into_chars(raw_data)
        if tokenized_data is None:
            assert False, "Could not process raw data into tokens."
        save_json(tokenized_data, tokens_file)
    else:
        print("Loaded tokenized data from file.")

    data_array = None
    if args["tokenize"] == "words":
        data_array = vectorize_sentences(args, tokenized_data, word2vec)
    else:
        data_array = vectorize_names(args, tokenized_data, word2vec)

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
    orig_data = raw_data[0].split(" ")
    orig_data_len = len(orig_data)
    for _ in range(10):
        index = random.randint(0, orig_data_len)
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
        index = random.randint(0, len(sent_decoded))
        outstr += output[index] + " "
    print outstr
    timestamp = int(time.time())
    filename = os.path.join(save_dir, "output_" + str(timestamp) + ".json")
    save_json(output, filename)
    """
    if args["tokenize"] == "chars":
        matches = 0
        duplicates = []
        dup_count = 0
        for o in output:
            if o not in duplicates:
                duplicates.append(0)
            else:
                dup_count += 1
            if o in original_data:
                matches += 1
        print("Found " + str(matches) + " matches between output and original data")
        print("Found " + str(dup_count) + " duplicates in output data")
    """

#########################################################
# Entry point
#########################################################
if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args = get_args()
    args["raw_data_file"] = os.path.join("data", "data.json")
    train_state_file = os.path.join(save_dir, "train_state.json")
    current_epoch = 0
    if os.path.exists(train_state_file):
        current_epoch = load_json(train_state_file)

    random.seed(1)
    train, word2vec, features, tokens = prepare_data(args)
    args["num_features"] = features
    batch_len = features * args["batch_size"]
    print
    print("Testing decoder")
    test_decoder(args, word2vec)
    print

    args["input_size"] = input_size = 1
    if args["tokenize"] == "chars":
        args["input_size"] = input_size = 1000
    args["original_dim"] = original_dim = len(train[0])
    epochs = args["num_epochs"]
    epochs_per_iter = args["sample_every"]
    if args["tokenize"] == "chars":
        epochs = epochs * 10

    intermediate_dim = int(original_dim / 3)
    latent_dim = int(intermediate_dim * 1.2)
    epsilon_std = args["epsilon_std"]
    magic = args["magic"]
    intermediate_activator = args["intermediate_activator"]
    optimizer = args["optimizer"]

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
                    verbose=0,
                    validation_data=(train, train),
                    callbacks=cp)
            current_epoch += epochs_per_iter
            save_json(current_epoch, train_state_file)
            sample(train, args, word2vec, encoder, generator)
    else:
        sample(train, args, word2vec, encoder, generator)


