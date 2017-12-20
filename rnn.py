# -*- coding: utf-8 -*-
from six.moves import cPickle
from text_handler import load_and_tokenize
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import numpy as np
import random
import argparse
import collections
import re
import io
import os
import json
import time
import sys


save_dir = "rnn_save"
data_dir = "data"
log_dir = "logs"
input_file = os.path.join(data_dir, "data.json")
tokens_file = os.path.join(save_dir, "tokens.json")
vocab_file = os.path.join(save_dir, "vocab.pkl")
vocab_json = os.path.join(save_dir, "vocab.json")
tensor_file = os.path.join(save_dir, "tensors.npy")
checkpoint_file = os.path.join(save_dir, "checkpoint")
train_state_file = os.path.join(save_dir, "train_state.json")
batch_pointer = 0

class Model():
    def __init__(self, args):
        self.args = args
        infer = False
        if args["mode"] == "sample":
            args["batch_size"] = 1
            args["seq_length"] = 1
            infer = True

        print("Initializing model")
        if args["model"] == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args["model"] == 'gru':
            cell_fn = rnn.GRUCell
        elif args["model"] == 'lstm':
            cell_fn = rnn.BasicLSTMCell

        cells = []
        for _ in range(args["num_layers"]):
            cell = cell_fn(args["rnn_size"])
            cells.append(cell)
        self.cell = cell = rnn.MultiRNNCell(cells)

        self.input_data = tf.placeholder(tf.int32, [args["batch_size"], args["seq_length"]])
        self.targets = tf.placeholder(tf.int32, [args["batch_size"], args["seq_length"]])
        self.initial_state = cell.zero_state(args["batch_size"], tf.float32)
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        tf.summary.scalar("time_batch", self.batch_time)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args["rnn_size"], args["vocab_size"]])
            variable_summaries(softmax_w)
            softmax_b = tf.get_variable("softmax_b", [args["vocab_size"]])
            variable_summaries(softmax_b)
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args["vocab_size"], args["rnn_size"]])
                inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), args["seq_length"], 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args["rnn_size"]])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args["batch_size"] * args["seq_length"]])],
                args["vocab_size"])
        self.cost = tf.reduce_sum(loss) / args["batch_size"] / args["seq_length"]
        tf.summary.scalar("cost", self.cost)
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args["grad_clip"])
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

def check_for_saved_state():
    saved = False
    if os.path.exists(vocab_file) and os.path.exists(tensor_file):
        saved = True
    return saved

def load_saved_state():
    print("Loading pre-saved state")
    vocab = None
    vocab_inv = None
    tensors = None
    print("Loading vocab")
    with open(vocab_file, 'rb') as f:
        vocab_inv = cPickle.load(f)
    vocab_size = len(vocab_inv)
    print("Vocab size: " + str(vocab_size))
    vocab = dict(zip(vocab_inv, range(len(vocab_inv))))
    print("Loading tensors")
    tensors = np.load(tensor_file)
    return vocab, vocab_inv, tensors

def build_vocab(tokens):
    print("Building vocab")
    element_counts = collections.Counter(tokens)
    print("Saving word counts.")
    with open("rnn_save/word_counts.json", "w") as file:
        json.dump(dict(element_counts), file, indent=4)
    vocab_inv = [x[0] for x in element_counts.most_common()]
    vocab_inv = list(sorted(vocab_inv))
    vocab = {x: i for i, x in enumerate(vocab_inv)}
    return [vocab, vocab_inv]

def process_input_data(input_file, tokenize):
    print("Processing input data")
    tokens = load_and_tokenize(input_file, tokenize)
    with open(tokens_file, "w") as f:
        json.dump(tokens, f, indent=4)
    vocab, vocab_inv = build_vocab(tokens)
    vocab_size = len(vocab_inv)
    print("Vocab size: " + str(vocab_size))
    tensors = np.array(list(map(vocab.get, tokens)))
    print("Saving vocab_inv as pickle")
    with open(vocab_file, 'wb') as f:
        cPickle.dump(vocab_inv, f)
    print("Saving vocab as json")
    with open(vocab_json, "w") as f:
        json.dump(vocab, f, indent=4)
    print("Saving tensor")
    np.save(tensor_file, tensors)
    return vocab, vocab_inv, tensors

def create_batches(tensors, batch_size, seq_length):
    print("Creating batches")
    num_batches = int(tensors.size / (batch_size * seq_length))
    if num_batches==0:
        assert False, "Not enough data. Make seq_length and batch_size smaller."

    tensors = tensors[:num_batches * batch_size * seq_length]
    xdata = tensors
    ydata = np.copy(tensors)

    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    x_batches = np.split(xdata.reshape(batch_size, -1), num_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), num_batches, 1)
    return x_batches, y_batches

def next_batch(x_batches, y_batches):
    global batch_pointer
    x, y = x_batches[batch_pointer], y_batches[batch_pointer]
    batch_pointer += 1
    return x, y

def generate_text(args):
    print("Sampling...")
    saved_state = check_for_saved_state()
    if saved_state == False:
        assert False, "No previous saved state to load for text generation."
    if not os.path.exists(checkpoint_file):
        assert False, "No previous checkpoint to load for text generation."
    vocab, vocab_inv, tensors = load_saved_state()
    args["vocab_size"] = len(vocab_inv)
    model = Model(args)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ret = ""
            next_word = ""
            print("Restoring saved model.")
            saver.restore(sess, ckpt.model_checkpoint_path)
            if os.path.exists(train_state_file):
                saved_epoch_pointer, saved_batch_pointer = load_trainer_state()
            ep = saved_epoch_pointer
            print("Model was trained for " + str(ep) + " epochs.")
            found = False
            next_word = ""
            while found != True:
                next_word = random.choice(list(vocab.keys()))
                if next_word[0].isupper():
                    found = True
            print("Randomly chosen starting token: " + next_word)
            state = sess.run(model.cell.zero_state(1, tf.float32))
            ret = next_word
            word = next_word.split()[-1]
            word_count = 0
            sentence_count = 0
            new_sentence = False
            finished = False

            def weighted_pick(weights):
                t = np.cumsum(weights)
                s = np.sum(weights)
                return(int(np.searchsorted(t, np.random.rand(1)*s)))

            sentence_count = 0
            token_count = 0
            max_tokens = args["n"] * 20
            if args["tokenize"] == "chars":
                max_tokens = args["n"] * 100
            while finished == False:
                x = np.zeros((1, 1))
                x[0, 0] = vocab.get(word, 0)
                feed = {model.input_data: x, model.initial_state:state}
                [probs, state] = sess.run([model.probs, model.final_state], feed)
                p = probs[0]
                sample = weighted_pick(p)
                pred = vocab_inv[sample]
                token_count += 1
                if pred in [".", "?", "!"]:
                    sentence_count += 1
                    print("Sentence: " + str(sentence_count))
                if pred == ".":
                    if ret.endswith("."):
                        sentence_count -= 1
                        print("Sentence: " + str(sentence_count))
                ret += pred
                word = pred
                if sentence_count > args["n"]:
                    print("Hit max sentences")
                    finished = True
                if token_count > max_tokens:
                    print("Hit max tokens")
                    finished = True
            print(ret)

def save_trainer_state(epochs, batch_position):
    variable = [epochs, batch_position]
    with open(train_state_file, "w") as f:
        json.dump(variable, f)

def load_trainer_state():
    variable = []
    with open(train_state_file, "r") as f:
        variable = json.load(f)
    return variable

def train_model(args):
    global batch_pointer
    print("Training...")
    saved_state = check_for_saved_state()
    vocab = None
    vocab_inv = None
    tensors = None
    if saved_state == True:
        print("Loading pre-processed data from disk.")
        vocab, vocab_inv, tensors = load_saved_state()
    else:
        print("No previously saved data exists. Starting from scratch.")
        vocab, vocab_inv, tensors = process_input_data(input_file, args["tokenize"])
    args["vocab_size"] = len(vocab_inv)
    num_batches = int(tensors.size / (args["batch_size"] * args["seq_length"]))
    x_batches, y_batches = create_batches(tensors, args["batch_size"], args["seq_length"])
    restore_checkpoint = False
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if ckpt is not None:
        print("Tensorflow restore point found.")
        restore_checkpoint = True
    if os.path.exists(train_state_file):
        saved_epoch_pointer, saved_batch_pointer = load_trainer_state()
    model = Model(args)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir)
    starting_epoch = 0
    print("Starting session.")
    saved_batch_pointer = 0
    with tf.Session() as sess:
        train_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        if restore_checkpoint == True:
            print("Restoring checkpoint")
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Saved batch pointer: " + str(saved_batch_pointer))
            print("Saved epoch pointer: " + str(saved_epoch_pointer))
            starting_epoch = saved_epoch_pointer
        try:
            for e in range(starting_epoch, args["num_epochs"]):
                sess.run(tf.assign(model.lr, args["learning_rate"] * (args["decay_rate"] ** e)))
                state = sess.run(model.initial_state)
                batch_pointer = saved_batch_pointer
                saved_batch_pointer = 0
                speed = 0
                for b in range(batch_pointer, num_batches):
                    start = time.time()
                    x, y = next_batch(x_batches, y_batches)
                    feed = {model.input_data: x, model.targets: y, model.initial_state: state,
                            model.batch_time: speed}
                    summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state, model.train_op, model.inc_batch_pointer_op], feed)
                    train_writer.add_summary(summary, e * num_batches + b)
                    speed = time.time() - start
                    if (e * num_batches + b) % args["output_every"] == 0:
                        print(str(b) + "/" + str(num_batches) + " (epoch:" + str(e) + "/" + str(args["num_epochs"]) + "), (train_loss:" + str("%.3f"%train_loss) + ") (speed:" + str("%.3f"%speed) + ")")
                    if (e * num_batches + b) % args["save_every"] == 0 \
                            or (e==args["num_epochs"]-1 and b == num_batches-1):
                        checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                        print("Saving...")
                        saver.save(sess, checkpoint_path, global_step = e * num_batches + b)
                        save_trainer_state(e, batch_pointer)
                        print("Model saved to " + checkpoint_path)
        except KeyboardInterrupt:
            print("Keyboard interrupt.")
            print("Saving. Please wait.")
            checkpoint_path = os.path.join(save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step = e * num_batches + b)
            save_trainer_state(e, batch_pointer)
            print("model saved to {}".format(checkpoint_path))
        train_writer.close()

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
    parser.add_argument('-n', type=int, default=25,
                       help='The number of sentences to generate')
    parser.add_argument('--rnn_size', type=int, default=300,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=100,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='number of epochs')
    parser.add_argument('--output_every', type=int, default=10,
                       help='screen output frequency')
    parser.add_argument('--save_every', type=int, default=500,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
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
    if args["mode"] == "train":
        train_model(args)
    else:
        generate_text(args)
