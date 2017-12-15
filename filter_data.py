from text_handler import split_line_into_words
import numpy as np
import json
import pickle
import string
import re
import os

keywords = ["nerf", "Nerf", "NERF", "cry", "QQ", "qq", "whine", "Whine", "Cry", "CRY", "WHINE", "balance", "BALANCE", "buff", "Buff", "BUFF", "scrub", "noob", "newb", "n00b", "NOOB", "reroll", "quit", "wtf", "WTF", "outclassed", "bench", "Bench", "stupid", "STUPID", "skill", "broken", "dumpster", "bro "]
max_len = 500
data_dir = "data/"

def sanitize(s):
    s = s.replace(u'\u201c', u'"').replace(u'\u201d', u'"').replace(u'\u2018', u'\'').replace(u'\u2019', u'\'').replace(u'\u2013', u'-')
    s = re.sub("\s+", " ", s)
    return s

def tokenize(s):
    ret, lost = split_line_into_words(s)
    return ret

def make_vocab(tokens):
    vocab = []
    freq_dist = {}
    vocab.append("")
    for sentence in tokens:
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

def create_padded_vectors(tokenized, max_len, w2idx):
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

def create_continuous_vector(tokenized, num_tokens, w2idx):
    indices = []
    count = 0
    print("Expected number of tokens: " + str(num_tokens))
    vector = np.zeros([num_tokens], dtype=np.int32)
    for sentence in tokenized:
        for word in sentence:
            if word in w2idx:
                indices.append(w2idx[word])
                count += 1
    print("Vectorized " + str(count) + " tokens.")
    vector = np.array(indices)
    print("Vector size: " + str(len(vector)))
    return vector

def process_data():
    raw = []
    with open(data_dir + "raw_data.json", "r") as f:
        raw = json.load(f)

    filtered = []
    print("Filtering...")
    for item in raw:
        item = sanitize(item)
        if len(item) < max_len:
            interesting = False
            for k in keywords:
                if k in item:
                    interesting = True
            if interesting == True:
                if len(item) > 0:
                    filtered.append(item)

    print("Filtered output contains " + str(len(filtered)) + " lines.")
    with open(data_dir + "filtered_data.json", "w") as f:
        json.dump(filtered, f, indent=4)


    print("Tokenizing...")
    tokens = []
    total_tokens = 0
    max_sent_len = 0
    min_sent_len = 200
    for item in filtered:
        toks = tokenize(item)
        if len(toks) > 5:
            if len(toks) > max_sent_len:
                max_sent_len = len(toks)
            if len(toks) < min_sent_len:
                min_sent_len = len(toks)
            total_tokens += len(toks)
            tokens.append(toks)

    print("Total tokens: " + str(total_tokens))
    print("Maximum tokenized sentence length: " + str(max_sent_len))
    print("Minimum tokenized sentence length: " + str(min_sent_len))
    with open(data_dir + "tokens.json", "w") as f:
        json.dump(tokens, f, indent=4)


    print('Creating vocab...')
    idx2w, w2idx, freq_dist = make_vocab(tokens)
    with open(data_dir + "idx2w.json", "w") as f:
        json.dump(idx2w, f, indent=4)
    with open(data_dir + "w2idx.json", "w") as f:
        json.dump(w2idx, f, indent=4)
    with open(data_dir + "freq_dist.json", "w") as f:
        json.dump(freq_dist, f, indent=4)


    print("Creating vectors")
    vectors = create_continuous_vector(tokens, total_tokens, w2idx)
    print len(vectors)
    np.save(data_dir + "vectors.npy", vectors)

    limit = {
             "maxlen" : max_sent_len,
             "minlen" : min_sent_len,
            }

    metadata = {
                "w2idx" : w2idx,
                "idx2w" : idx2w,
                "limit" : limit,
                "freq_dist" : freq_dist
               }
    with open(data_dir + "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    return metadata, vectors

def load_data():
    loaded = False
    metadata = None
    idx_q = None
    idx_a = None
    print("Attempting to load pre-processed data")
    if os.path.exists(data_dir):
        if os.path.exists(data_dir + "metadata.pkl"):
            with open(data_dir + "metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
        if os.path.exists(data_dir + "idx_q.npy"):
            idx_q = np.load(data_dir + "idx_q.npy")
        if os.path.exists(data_dir + "idx_a.npy"):
            idx_a = np.load(data_dir + "idx_a.npy")
    else:
        os.makedirs(data_dir)
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
        return process_data()

if __name__ == '__main__':
    process_data()






