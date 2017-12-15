from text_handler import split_line_into_words
import numpy as np
import json
import pickle
import string
import re
import os

keywords = ["nerf", "Nerf", "NERF", "cry", "QQ", "qq", "whine", "Whine", "Cry", "CRY", "WHINE", "balance", "BALANCE", "buff", "Buff", "BUFF", "scrub", "noob", "newb", "n00b", "NOOB", "reroll", "quit", "wtf", "outclassed", "bench", "Bench", "stupid", "STUPID", "skill", "broken", "dumpster", "bro"]
max_len = 500
data_dir = "conv_data/"

def sanitize(s):
    s = s.replace(u'\u201c', u'"').replace(u'\u201d', u'"').replace(u'\u2018', u'\'').replace(u'\u2019', u'\'').replace(u'\u2013', u'-')
    s = re.sub("\s+", " ", s)
    return s

def tokenize(s):
    ret, lost = split_line_into_words(s)
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

def process_data():
    raw = []
    with open(data_dir + "conv.json", "r") as f:
        raw = json.load(f)

    filtered = []
    print("Filtering...")
    for item in raw:
        str1, str2 = item
        str1 = sanitize(str1)
        str2 = sanitize(str2)
        if len(str1) < max_len and len(str2) < max_len:
            interesting = False
            for k in keywords:
                if k in str1 or k in str2:
                    interesting = True
            if interesting == True:
                if len(str1) > 0 and len(str2) > 0:
                    filtered.append([str1, str2])

    print("Filtered output contains " + str(len(filtered)) + " lines.")
    with open(data_dir + "filtered_conv.json", "w") as f:
        json.dump(filtered, f, indent=4)


    print("Tokenizing...")
    qlist = []
    alist = []
    max_sent_len = 0
    min_sent_len = 200
    for item in filtered:
        qstr, astr = item
        qtok = tokenize(qstr)
        atok = tokenize(astr)
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
    with open(data_dir + "qtokens.json", "w") as f:
        json.dump(qlist, f, indent=4)
    with open(data_dir + "atokens.json", "w") as f:
        json.dump(alist, f, indent=4)


    print('Creating vocab...')
    idx2w, w2idx, freq_dist = make_vocab(qlist, alist)
    with open(data_dir + "idx2w.json", "w") as f:
        json.dump(idx2w, f, indent=4)
    with open(data_dir + "w2idx.json", "w") as f:
        json.dump(w2idx, f, indent=4)
    with open(data_dir + "freq_dist.json", "w") as f:
        json.dump(freq_dist, f, indent=4)


    print("Creating vectors")
    qvectors = create_vectors(qlist, max_sent_len, w2idx)
    print len(qvectors)
    np.save(data_dir + "idx_q.npy", qvectors)

    avectors = create_vectors(alist, max_sent_len, w2idx)
    print len(avectors)
    np.save(data_dir + "idx_a.npy", avectors)

    limit = {
             "maxq" : max_sent_len,
             "minq" : min_sent_len,
             "maxa" : max_sent_len,
             "mina" : min_sent_len,
            }

    metadata = {
                "w2idx" : w2idx,
                "idx2w" : idx2w,
                "limit" : limit,
                "freq_dist" : freq_dist
               }
    with open(data_dir + "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    return metadata, qvectors, avectors

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

def split_dataset(x, y, ratio = [0.7, 0.15, 0.15] ):
    # number of examples
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)

def batch_gen(x, y, batch_size):
    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T

def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T

def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])

if __name__ == '__main__':
    process_data()






