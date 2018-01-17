# -*- coding: utf8 -*-
from text_handler import split_line_into_words, split_line_into_chars, sanitize_line
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import argparse
import json
import sys
import time
import os
import pickle
import string
import re

save_dir = "seq2seq_save"
seed = "I don't know... you seem like the one QQing."

def tokenize(s, mode):
    ret = []
    if mode == "words":
        ret, lost = split_line_into_words(s)
    else:
        ret = split_line_into_chars(s)
    return ret

def load_data(args):
    loaded = False
    metadata = None
    idx_q = None
    idx_a = None
    print("Attempting to load pre-processed data")
    metadata_file = os.path.join(save_dir, "metadata.pkl")
    qvector_file = os.path.join(save_dir, "idx_q.npy")
    avector_file = os.path.join(save_dir, "idx_a.npy")
    if os.path.exists(save_dir):
        if os.path.exists(metadata_file):
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
        if os.path.exists(qvector_file):
            idx_q = np.load(qvector_file)
        if os.path.exists(avector_file):
            idx_a = np.load(avector_file)
    else:
        os.makedirs(save_dir)
    if metadata is not None and idx_q is not None and idx_a is not None:
        if len(idx_q) > 0 and len(idx_a) > 0:
            if len(idx_q) == len(idx_a):
                loaded = True
            else:
                print("Vectors weren't of the same length.")
                print len(idx_q)
                print len(idx_a)
    if loaded == True:
        print("Successfully loaded pre-processed data.")
        return metadata, idx_q, idx_a
    else:
        print("Failed to load save state")
        sys.exit(0)

def prepare_data(args):
    print("Attemping to load data")
    metadata, trainX, trainY = load_data(args)
    assert len(trainX) == len(trainY)

    trainX = trainX.tolist()
    trainY = trainY.tolist()

    trainX = tl.prepro.remove_pad_sequences(trainX)
    trainY = tl.prepro.remove_pad_sequences(trainY)

    training_set = {}
    training_set["trainX"] = trainX
    training_set["trainY"] = trainY

    max_len = metadata["max_len"]
    idx2w = metadata['idx2w']
    w2idx = metadata['w2idx']
    start_id = len(idx2w)
    end_id = len(idx2w) + 1

    w2idx.update({'start_id': start_id})
    w2idx.update({'end_id': end_id})
    idx2w = idx2w + ['start_id', 'end_id']
    vocab = {}
    vocab["idx2w"] = idx2w
    vocab["w2idx"] = w2idx

    params = {}
    params["max_len"] = max_len
    params["start_id"] = start_id
    params["end_id"] = end_id
    params["pad_id"] = w2idx['']

    return params, vocab, training_set


def model(encode_seqs, decode_seqs, args, mode):
    dropout = 0.5
    reuse = False
    vocab_size = args["vocab_size"]
    rnn_size = args["rnn_size"]
    num_layers = args["num_layers"]
    if mode == "sample":
        dropout = None
        reuse = True
    with tf.variable_scope("model", reuse=reuse):
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(inputs = encode_seqs,
                                             vocabulary_size = vocab_size,
                                             embedding_size = rnn_size,
                                             name = 'seq_embedding')
            vs.reuse_variables()
            tl.layers.set_name_reuse(True)
            net_decode = EmbeddingInputlayer(inputs = decode_seqs,
                                             vocabulary_size = vocab_size,
                                             embedding_size = rnn_size,
                                             name = 'seq_embedding')
        net_rnn = Seq2Seq(net_encode,
                          net_decode,
                          cell_fn = tf.contrib.rnn.BasicLSTMCell,
                          n_hidden = rnn_size,
                          initializer = tf.random_uniform_initializer(-0.1, 0.1),
                          encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                          decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                          initial_state_encode = None,
                          dropout = dropout,
                          n_layer = num_layers,
                          return_seq_2d = True,
                          name = 'seq2seq')
        net_out = DenseLayer(net_rnn,
                             n_units=vocab_size,
                             act=tf.identity,
                             name='output')
    return net_out, net_rnn

def sample(args, context, sess):
    w2idx = context["w2idx"]
    idx2w = context["idx2w"]
    end_id = args["end_id"]
    new_seed = seed
    filename = os.path.join(save_dir, "sample_out.txt")
    handle = open(filename, "a")
    users = ["n00bk1ller", "ganksquad1"]
    current_user = 0
    while True:
        sys.stdout.write(users[current_user] + "> ")
        sys.stdout.flush()
        type_sentence(new_seed)
        print
        print
        handle.write(new_seed + "\n\n")
        sanitized = sanitize_line(new_seed)
        tokens = tokenize(sanitized, args["tokenize"])
        seed_id = []
        for w in tokens:
            if w in w2idx:
                seed_id.append(w2idx[w])
        for _ in range(args["n"]):
            state = sess.run(context["net_rnn"].final_state_encode,
                            {encode_seqs2: [seed_id]})
            o, state = sess.run([context["y"], context["net_rnn"].final_state_decode],
                            {context["net_rnn"].initial_state_decode: state,
                            decode_seqs2: [[args["start_id"]]]})
            w_id = tl.nlp.sample_top(o[0], top_k=3)
            w = idx2w[w_id]
            sentence = [w]
            for _ in range(args["max_len"]):
                o, state = sess.run([context["y"], context["net_rnn"].final_state_decode],
                                {context["net_rnn"].initial_state_decode: state,
                                decode_seqs2: [[w_id]]})
                w_id = tl.nlp.sample_top(o[0], top_k=2)
                w = idx2w[w_id]
                if w_id == end_id:
                    break
                sentence = sentence + [w]
            sentence = "".join(sentence)
            new_seed = sentence
            if current_user == 0:
                current_user = 1
            else:
                current_user = 0

def type_sentence(sentence):
    for c in list(sentence):
        sys.stdout.write(c)
        sys.stdout.flush()
        if c == " ":
            time.sleep(float(random.randint(100,230)/1000.0))
        elif c in ".,!?/-+_":
            time.sleep(float(random.randint(130,350)/1000.0))
        elif c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            time.sleep(float(random.randint(60,180)/1000.0))
        else:
            time.sleep(float(random.randint(20,150)/1000.0))

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
    parser.add_argument('--tokenize', type=str, default='chars',
                       help='tokenize by words or chars')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='minibatch size')
    parser.add_argument('--rnn_size', type=int, default=1024,
                       help='number of neurons')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='number of layers')
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='learning rate')
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
    print("Args:")
    for key, value in sorted(args.iteritems()):
        print("\t" + str(key) + ":\t" + str(value))
    save_args(args)
    return args

if __name__ == '__main__':
    if not os.path.exists(save_dir):
        print("Creating save directory: " + save_dir)
        os.makedirs(save_dir)
    args = get_args()

    params, vocab, training_set = prepare_data(args)

    save_file = os.path.join(save_dir, "n.npz")

    args["vocab_size"] = len(vocab["idx2w"])
    args["max_len"] = params["max_len"]
    args["start_id"] = params["start_id"]
    args["end_id"] = params["end_id"]
    args["pad_id"] = params["pad_id"]
    args["n_step"] = int(len(training_set["trainX"])/args["batch_size"])
    save_args(args)

    context = {}
    context["w2idx"] = vocab["w2idx"]
    context["idx2w"] = vocab["idx2w"]
    encode_seqs = tf.placeholder(dtype=tf.int64,
                                 shape=[args["batch_size"], None], 
                                 name="encode_seqs")
    decode_seqs = tf.placeholder(dtype=tf.int64,
                                 shape=[args["batch_size"], None],
                                 name="decode_seqs")
    target_seqs = tf.placeholder(dtype=tf.int64,
                                 shape=[args["batch_size"], None],
                                 name="target_seqs")
    target_mask = tf.placeholder(dtype=tf.int64,
                                 shape=[args["batch_size"], None],
                                 name="target_mask")
    context["net_out"], _ = model(encode_seqs, decode_seqs, args, "train")
    context["loss"] = tl.cost.cross_entropy_seq_with_mask(logits=context["net_out"].outputs, 
                                               target_seqs=target_seqs, 
                                               input_mask=target_mask,
                                               return_details=False,
                                               name='cost')
    context["train_op"] = tf.train.AdamOptimizer(learning_rate=args["learning_rate"]).minimize(context["loss"])
    context["net_out"].print_params(False)

    encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
    decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
    context["net"], context["net_rnn"] = model(encode_seqs2, decode_seqs2, args, "sample")
    context["y"] = tf.nn.softmax(context["net"].outputs)

    print
    print("Session init...")
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if os.path.exists(save_file):
        print("Restoring previous save.")
        tl.files.load_and_assign_npz(sess=sess, name=save_file, network=context["net"])
        print("Done restoring.")
    else:
        print("No previous save file found.")
        sys.exit(0)

    print("Chatbot mode engaged.")
    sample(args, context, sess)

