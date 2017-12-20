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
import time
import os
import pickle
import string
import re

save_dir = "seq2seq_save"
data_dir = "conv_data"
seeds = ["All i see with rogues is negativity, they all say how rogue is dead in legion."]
seeds_full = ["All i see with rogues is negativity, they all say how rogue is dead in legion.",
        "I don't know... you seem like the one QQing.",
        "There is no point in people like OP's friends playing anyway because they will just QQ about everything by the sounds of it.",
        "No no, that would have cost money .... they pulled some scrub intern for the job and had it done for free because they don't know any better",
        "Yeah, every spec is raid viable. That's why ret, beast mastery hunters, and assassination rogues are benched. That's also why ret has spent the entire expansion warming the bench.",
        "I miss my paladin. Also, yeah I got benched for a bear and a monk.",
        "What do you do that takes any skill? Keep your QQ'n up it's working."]

keywords = ["nerf", "Nerf", "NERF", "cry", "QQ", "qq", "whine", "Whine", "Cry", "CRY", "WHINE", "balance", "BALANCE", "buff", "Buff", "BUFF", "scrub", "noob", "newb", "n00b", "NOOB", "reroll", "quit", "wtf", "outclassed", "bench", "Bench", "stupid", "STUPID", "skill", "broken", "dumpster", "bro"]
max_len = 500

def tokenize(s, mode):
    ret = []
    if mode == "words":
        ret, lost = split_line_into_words(s)
    else:
        ret = split_line_into_chars(s)
    return ret

def make_vocab(qtokens, atokens):
    vocab = []
    freq_dist = {}
    vocab.append("")
    for sentence in qtokens:
        for word in sentence:
            if word not in vocab:
                vocab.append(word)
            if word not in freq_dist:
                freq_dist[word] = 1
            else:
                freq_dist[word] += 1
    for sentence in atokens:
        for word in sentence:
            if word not in vocab:
                vocab.append(word)
            if word not in freq_dist:
                freq_dist[word] = 1
            else:
                freq_dist[word] += 1
    print("Vocab size: " + str(len(vocab)))
    index2word = vocab
    word2index = {word: index for index, word in enumerate(index2word)}
    return index2word, word2index, freq_dist

def create_vectors(tokenized, max_len, w2idx):
    num_rows = len(tokenized)
    vector = np.zeros([num_rows, max_len], dtype=np.int32)
    for index, sentence in enumerate(tokenized):
        indices = []
        for word in sentence:
            if word in w2idx:
                indices.append(w2idx[word])
        indices += [0] * (max_len - len(sentence))
        vector[index] = np.array(indices)
    return vector

def process_data(args):
    raw = []
    raw_data_file = os.path.join(data_dir, "conv.json")
    with open(raw_data_file, "r") as f:
        raw = json.load(f)

    max_len = args["sentence_len"]
    tokenize_mode = args["tokenize"]
    filtered = []
    print("Filtering...")
    for item in raw:
        str1, str2 = item
        str1 = sanitize_line(str1)
        str2 = sanitize_line(str2)
        if len(str1) < max_len and len(str2) < max_len:
            interesting = False
            for k in keywords:
                if k in str1 or k in str2:
                    interesting = True
            if interesting == True:
                if len(str1) > 0 and len(str2) > 0:
                    filtered.append([str1, str2])

    print("Filtered output contains " + str(len(filtered)) + " lines.")
    filtered_data_file = os.path.join(save_dir, "filtered_conv.json")
    with open(filtered_data_file, "w") as f:
        json.dump(filtered, f, indent=4)


    print("Tokenizing...")
    qlist = []
    alist = []
    max_sent_len = 0
    min_sent_len = 200
    for item in filtered:
        qstr, astr = item
        qtok = tokenize(qstr, tokenize_mode)
        atok = tokenize(astr, tokenize_mode)
        if len(qtok) > 5 and len(atok) > 5:
            if len(qtok) > max_sent_len:
                max_sent_len = len(qtok)
            if len(atok) > max_sent_len:
                max_sent_len = len(atok)
            if len(qtok) < min_sent_len:
                min_sent_len = len(qtok)
            if len(atok) < min_sent_len:
                min_sent_len = len(atok)
            qlist.append(qtok)
            alist.append(atok)

    print("Maximum tokenized sentence length: " + str(max_sent_len))
    print("Minimum tokenized sentence length: " + str(min_sent_len))
    qtokens_file = os.path.join(save_dir, "qtokens.json")
    with open(qtokens_file, "w") as f:
        json.dump(qlist, f, indent=4)
    atokens_file = os.path.join(save_dir, "atokens.json")
    with open(atokens_file, "w") as f:
        json.dump(alist, f, indent=4)


    print('Creating vocab...')
    idx2w, w2idx, freq_dist = make_vocab(qlist, alist)
    idx2w_file = os.path.join(save_dir, "idx2w.json")
    with open(idx2w_file, "w") as f:
        json.dump(idx2w, f, indent=4)
    w2idx_file = os.path.join(save_dir, "w2idx.json")
    with open(w2idx_file, "w") as f:
        json.dump(w2idx, f, indent=4)
    freq_dist_file = os.path.join(save_dir, "freq_dist.json")
    with open(freq_dist_file, "w") as f:
        json.dump(freq_dist, f, indent=4)


    print("Creating vectors")
    qvectors = create_vectors(qlist, max_sent_len, w2idx)
    print len(qvectors)
    qvector_file = os.path.join(save_dir, "idx_q.npy")
    np.save(qvector_file, qvectors)

    avectors = create_vectors(alist, max_sent_len, w2idx)
    print len(avectors)
    avector_file = os.path.join(save_dir, "idx_a.npy")
    np.save(avector_file, avectors)

    metadata = {
                "w2idx" : w2idx,
                "idx2w" : idx2w,
                "max_len" : max_sent_len,
                "min_len" : min_sent_len,
                "freq_dist" : freq_dist
               }
    metadata_file = os.path.join(save_dir, "metadata.pkl")
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

    return metadata, qvectors, avectors

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
        print("Pre-processed data didn't load. Processing data now.")
        return process_data(args)

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

    print("encode_seqs", [idx2w[id] for id in trainX[10]])
    target_seqs = tl.prepro.sequences_add_end_id([trainY[10]], end_id=end_id)[0]
    print("target_seqs", [idx2w[id] for id in target_seqs])
    decode_seqs = tl.prepro.sequences_add_start_id([trainY[10]], start_id=start_id, remove_last=False)[0]
    print("decode_seqs", [idx2w[id] for id in decode_seqs])
    target_mask = tl.prepro.sequences_get_mask([target_seqs])[0]
    print("target_mask", target_mask)
    print(len(target_seqs), len(decode_seqs), len(target_mask))
    exit

    return params, vocab, training_set


def view_data():
    for i in range(len(X)):
         print(i, [idx2w[id] for id in X[i]])
         print(i, [idx2w[id] for id in Y[i]])
         print(i, [idx2w[id] for id in _target_seqs[i]])
         print(i, [idx2w[id] for id in _decode_seqs[i]])
         print(i, _target_mask[i])
         print(len(_target_seqs[i]), len(_decode_seqs[i]), len(_target_mask[i]))
    exit()

def model(encode_seqs, decode_seqs, args, mode):
    dropout = 0.5
    reuse = False
    vocab_size = args["vocab_size"]
    emb_dim = args["emb_dim"]
    if mode == "sample":
        dropout = None
        reuse = True
    with tf.variable_scope("model", reuse=reuse):
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(inputs = encode_seqs,
                                             vocabulary_size = vocab_size,
                                             embedding_size = emb_dim,
                                             name = 'seq_embedding')
            vs.reuse_variables()
            tl.layers.set_name_reuse(True)
            net_decode = EmbeddingInputlayer(inputs = decode_seqs,
                                             vocabulary_size = vocab_size,
                                             embedding_size = emb_dim,
                                             name = 'seq_embedding')
        net_rnn = Seq2Seq(net_encode,
                          net_decode,
                          cell_fn = tf.contrib.rnn.BasicLSTMCell,
                          n_hidden = emb_dim,
                          initializer = tf.random_uniform_initializer(-0.1, 0.1),
                          encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                          decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                          initial_state_encode = None,
                          dropout = dropout,
                          n_layer = 3,
                          return_seq_2d = True,
                          name = 'seq2seq')
        net_out = DenseLayer(net_rnn,
                             n_units=vocab_size,
                             act=tf.identity,
                             name='output')
    return net_out, net_rnn

def train(args, training_set, context, sess):
    for epoch in range(args["current_epoch"], args["num_epochs"]):
        epoch_time = time.time()
        trainX, trainY = shuffle(training_set["trainX"], training_set["trainY"], random_state=0)
        total_err, n_iter = 0, 0
        print("Starting epoch: " + str(epoch))
        for X, Y in tl.iterate.minibatches(inputs=trainX,
                                           targets=trainY,
                                           batch_size=args["batch_size"],
                                           shuffle=False):
            step_time = time.time()

            X = tl.prepro.pad_sequences(X)
            _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=args["end_id"])
            _target_seqs = tl.prepro.pad_sequences(_target_seqs)

            _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=args["start_id"], remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

            _, err = sess.run([context["train_op"], context["loss"]],
                            {encode_seqs: X,
                            decode_seqs: _decode_seqs,
                            target_seqs: _target_seqs,
                            target_mask: _target_mask})

            if n_iter % args["output_every"] == 0:
                print("Epoch[%d/%d] step:[%d/%d] loss:%f took:%.5fs" % (epoch, args["num_epochs"], n_iter, args["n_step"], err, time.time() - step_time))

            total_err += err; n_iter += 1
            if n_iter % args["save_every"] == 0:
                sample(args, context, sess)
                print("Saving...")
                tl.files.save_npz(context["net"].all_params, name=save_file, sess=sess)
                save_trainer_state(epoch, n_iter)
        print
        print("New epoch.")
        print("Epoch[%d/%d] averaged loss:%f took:%.5fs" % (epoch, args["num_epochs"], total_err/n_iter, time.time()-epoch_time))
        print("Saving...")
        tl.files.save_npz(context["net"].all_params, name=save_file, sess=sess)
        save_trainer_state(epoch, n_iter)

def sample(args, context, sess):
    w2idx = context["w2idx"]
    idx2w = context["idx2w"]
    end_id = args["end_id"]
    for seed in seeds:
        print("Query >", seed)
        sanitized = sanitize(seed)
        tokens = tokenize(sanitized)
        seed_id = [w2idx[w] for w in tokens]
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
            print(" >", "".join(sentence))

def save_trainer_state(epochs, batch_position):
    variable = [epochs, batch_position]
    with open(train_state_file, "w") as f:
        json.dump(variable, f)

def load_trainer_state():
    variable = []
    with open(train_state_file, "r") as f:
        variable = json.load(f)
    return variable

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
    parser.add_argument('-n', type=int, default=1,
                       help='The number of sentences to generate')
    parser.add_argument('--sentence_len', type=int, default=500,
                       help='maximum sentence length')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='number of epochs')
    parser.add_argument('--output_every', type=int, default=1,
                       help='screen output frequency')
    parser.add_argument('--save_every', type=int, default=10,
                       help='save frequency')
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
    args["mode"] = cl_args["mode"]
    args["n"] = cl_args["n"]
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

    current_epoch = 0
    current_batch = 0
    save_file = os.path.join(save_dir, "n.npz")
    train_state_file = os.path.join(save_dir, "train_state.json")
    if os.path.exists(train_state_file):
        current_epoch, current_batch = load_trainer_state()

    args["current_batch"] = current_batch
    args["current_epoch"] = current_epoch
    args["vocab_size"] = len(vocab["idx2w"])
    args["max_len"] = params["max_len"]
    args["start_id"] = params["start_id"]
    args["end_id"] = params["end_id"]
    args["pad_id"] = params["pad_id"]
    args["n_step"] = int(len(training_set["trainX"])/args["batch_size"])
    args["emb_dim"] = 1024
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

    print("Session init...")
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if os.path.exists(save_file):
        print("Restoring previous save.")
        tl.files.load_and_assign_npz(sess=sess, name=save_file, network=net)
    else:
        print("No previous save file found.")

    if args["mode"] == "train":
        print("Training mode engaged.")
        train(args, training_set, context, sess)
    else:
        print("Sampling mode engaged.")
        sample(args, context, sess)

