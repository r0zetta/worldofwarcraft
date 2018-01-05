from six.moves import cPickle
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
from gensim import corpora, models, similarities 
import gensim.models.word2vec as w2v
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from text_handler import split_line_into_words
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import logging
import multiprocessing
import os
import nltk
import json
import sys

#plot_lims = ["xmin":-15, "xmax":15, "ymin":15, "ymax":15]
#test_words = ["trump", "bannon", "war", "nuclear", "iran", "america", "russia", "korea", "impeach", "hillary", "god", "fbi"]

input_files = ["battle_net_data/data.json", "mmo_champion_data/data.json"]
test_groups = [["sylvanas", "horde"], ["nerf", "buff"], ["affliction", "nerf"], ["pvp", "gank"], ["alliance", "horde"], ["raid", "raiding"], ["anduin", "sylvanas"], ["warrior", "mage", "priest"]]
test_words = ["illidan", "sylvanas", "anduin", "nerf", "buff", "warrior", "priest", "mage", "elf", "void", "pvp", "gank", "raid", "raiding", "nighthold", "tomb", "antorus", "varimathras", "argus", "coven", "affliction", "lock", "shadow", "alliance", "horde", "evil", "nice", "reroll", "quit", "lol", "qq", "bench", "wtf", "broken", "noob", "hunter"]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
save_dir = "w2v_cluster"
num_similar = 40

def print_progress():
    sys.stdout.write("#")
    sys.stdout.flush()

def save_bin(item, filename):
    with open(filename, "wb") as f:
        cPickle.dump(item, f)

def load_bin(filename):
    ret = None
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            ret = cPickle.load(f)
    return ret

def save_json(variable, filename):
    with open(filename, "w") as f:
        json.dump(variable, f, indent=4)

def load_json(filename):
    ret = None
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                ret = json.load(f)
        except:
            print("Couldn't load " + filename + ".")
    else:
        print(filename + " didn't exist.")
    return ret

def get_saved_args():
    filename = os.path.join(save_dir, "args.json")
    saved = load_json(filename)
    return saved

def save_args(args):
    filename = os.path.join(save_dir, "args.json")
    save_json(args, filename)

def get_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                       help='train or sample')
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
    print("Args:")
    for key, value in sorted(args.iteritems()):
        print("\t" + str(key) + ":\t" + str(value))
    return args

def split_into_sentences(raw_data):
    nltk.download("punkt")
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(raw_data)
    num_raw_sentences = len(raw_sentences)
    print("Raw sentence count: " + str(num_raw_sentences))
    return raw_sentences

def tokenize_sentences(sentences):
    ret = []
    max_s = len(sentences)
    prog_at = max_s / 50
    for count, s in enumerate(sentences):
        if len(s) > 0:
            tokens, lost = split_line_into_words(s)
            if len(tokens) > 0:
                ret.append(tokens)
        if count % prog_at == 0:
            print_progress()
    return ret

def clean_sentences(tokens, stopwords, stemmer):
    ret = []
    max_s = len(tokens)
    prog_at = max_s / 50
    for count, sentence in enumerate(tokens):
        if count % prog_at == 0:
            print_progress()
        cleaned = []
        for token in sentence:
            if len(token) > 0:
                if re.search("^\w+$", token):
                    token = token.lower()
                    skip = False
                    if skip == False:
                        if stopwords is not None:
                            for s in stopwords:
                                if token == s:
                                    #print("Removed stopword " + token)
                                    skip = True
                                    token = None
                                    break
                    if skip == False:
                        if re.search("[0-9]+", token):
                            skip = True
                            token = None
                    if skip == False:
                        if stemmer is not None:
                            stem = stemmer.stem(token)
                            if stem is not None:
                                #print("Stemmed " + token + " to " + stem)
                                token = stem
                    if skip == False:
                        if token is not None:
                            cleaned.append(token)
        if len(cleaned) > 0:
            ret.append(cleaned)
    return ret

def load_raw_data(input_files):
    ret = ""
    for x in input_files:
        if os.path.exists(x):
            raw = load_json(x)
            ret += "\n".join(raw)
    return ret

def prepare_data():
    print("Loading and tokenizing raw data")
    tokens_file = os.path.join(save_dir, "tokens.json")
    tokens = load_json(tokens_file)
    if tokens is None:
        print("Loading raw data")
        raw_data = load_raw_data(input_files)

        print("Splitting data into sentences")
        raw_sentences = split_into_sentences(raw_data)
        print("Tokenizing sentences")
        tokens = tokenize_sentences(raw_sentences)
        save_json(tokens, tokens_file)

    sentence_count = len(tokens)
    print("[TOKENS] Number of sentences: " + str(sentence_count))
    token_count = sum([len(sentence) for sentence in tokens])
    print("[TOKENS] The corpus contains " + str(token_count) + " tokens")

    print("Cleaning and stemming tokens")
    cleaned_file = os.path.join(save_dir, "cleaned.json")
    cleaned = load_json(cleaned_file)
    if cleaned is None:
        stopwords_file = "data/stopwords-iso.json"
        stopwords = load_json(stopwords_file)
        stopwords_en = None
        if stopwords is not None:
            stopwords_en = stopwords["en"]
        stemmer = None
        #stemmer = SnowballStemmer("english")
        cleaned1 = clean_sentences(tokens, stopwords_en, stemmer)
        cleaned = []
        for sent in cleaned1:
            if len(sent) > 0:
                cleaned.append(sent)
            else:
                print("Found empty sentence. WTF??!?!?!")
        save_json(cleaned, cleaned_file)
    sentence_count = len(cleaned)
    print("[CLEANED] Number of sentences: " + str(sentence_count))
    token_count = sum([len(sentence) for sentence in cleaned])
    print("[CLEANED] The corpus contains " + str(token_count) + " tokens")

    return tokens, cleaned

def LDA(cleaned):
    model_save_name = os.path.join(save_dir, "lda_model.sav")
    lda = None
    if os.path.exists(model_save_name):
        print("Loading model from disk")
        lda = models.LdaModel.load(model_save_name)
    else:
        tokens = []
        print("Creating dictionary")
        dictionary = corpora.Dictionary(cleaned)

        full_dict_items = dictionary.items()
        filename = os.path.join(save_dir, "LDA_full_dictionary.json")
        save_json(full_dict_items, filename)
        full_vocab = []
        for item in full_dict_items:
            full_vocab.append(item[1])
        print("Full vocab contained " + str(len(full_vocab)) + " items.")
        filename = os.path.join(save_dir, "LDA_full_vocab.json")
        save_json(full_vocab, filename)
        print("Creating corpus")
        count = 0
        max_i = len(cleaned)
        prog = max_i/50
        for sent in cleaned:
            tokens += sent
        corpus = [dictionary.doc2bow(tokens, return_missing=False)]
        print("Building model")
        lda = models.LdaModel(corpus,
                              num_topics=10,
                              id2word=dictionary,
                              distributed=False,
                              chunksize=2000,
                              passes=10,
                              update_every=1,
                              alpha='symmetric',
                              eta=None,
                              decay=0.5,
                              offset=1.0,
                              eval_every=None,
                              iterations=50,
                              gamma_threshold=0.001,
                              minimum_probability=0.01,
                              random_state=1,
                              ns_conf=None,
                              minimum_phi_value=0.01,
                              per_word_topics=True,
                              callbacks=None)
        lda.save(model_save_name)

    topics_matrix = lda.show_topics(formatted=True, num_words=5)
    topics_save_name = os.path.join(save_dir, "topics_matrix.json")
    with open(topics_save_name, "w") as f:
        json.dump(topics_matrix, f, indent=4)

def k_means_cluster(word2vec, num_clusters):
    word_vectors = word2vec.wv.syn0
    print("Number of word vectors: " + str(len(word_vectors)))
    model = KMeans(n_clusters=num_clusters,
                   init='k-means++',
                   max_iter=100,
                   n_init=1,
                   random_state=1,
                   verbose=1)
    idx = model.fit_predict(word_vectors)
    word_centroid_map = dict(zip(word2vec.wv.index2word, idx ))
    max_i = len(word_centroid_map.values())
    progress_point = max_i / 50
    print("Calculating clusters over " + str(max_i) + " words.")
    words = []
    for cl in range(num_clusters):
        cluster = []
        for i in range(max_i):
            if i % progress_point == 0:
                print_progress()
            if(word_centroid_map.values()[i] == cl):
                cluster.append(word_centroid_map.keys()[i])
        words.append(cluster)
        print("\nCluster " + str(cl) + " had " + str(len(cluster)) + " items.")
    filename = os.path.join(save_dir, "clusters.json")
    save_json(words, filename)
    return words

def get_word_frequencies(corpus):
    print("Calculating word frequencies.")
    frequencies = Counter()
    for sent in corpus:
        for word in sent:
            frequencies[word] += 1
    freq = frequencies.most_common()
    return freq

def get_word2vec(sentences):
    num_workers = multiprocessing.cpu_count()
    num_features = 300
    epoch_count = 1000
    sentence_count = len(sentences)
    w2v_file = os.path.join(save_dir, "word_vectors.w2v")
    word2vec = None
    if os.path.exists(w2v_file):
        print("w2v model loaded from " + w2v_file)
        word2vec = w2v.Word2Vec.load(w2v_file)
    else:
        word2vec = w2v.Word2Vec(sg=1,
                                seed=1,
                                workers=num_workers,
                                size=num_features,
                                min_count=3,
                                window=7,
                                sample=0)

        print("Building vocab...")
        word2vec.build_vocab(sentences)
        print("Word2Vec vocabulary length:", len(word2vec.wv.vocab))
        print("Training...")
        word2vec.train(sentences, total_examples=sentence_count, epochs=epoch_count)
        print("Saving model...")
        word2vec.save(w2v_file)
    return word2vec

def create_embeddings(word2vec):
    print("Creating embeddings for tensorboard")
    all_word_vectors_matrix = word2vec.wv.syn0
    num_words = len(all_word_vectors_matrix)
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    dim = word2vec.wv[vocab[0]].shape[0]
    embedding = np.empty((num_words, dim), dtype=np.float32)
    metadata = ""
    for i, word in enumerate(vocab):
        embedding[i] = word2vec.wv[word]
        metadata += word + "\n"
    metadata_file = os.path.join(save_dir, "metadata.tsv")
    with open(metadata_file, "w") as f:
        f.write(metadata)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    X = tf.Variable([0.0], name='embedding')
    place = tf.placeholder(tf.float32, shape=embedding.shape)
    set_x = tf.assign(X, place, validate_shape=False)
    sess.run(tf.global_variables_initializer())
    sess.run(set_x, feed_dict={place: embedding})

    summary_writer = tf.summary.FileWriter(save_dir, sess.graph)
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = 'embedding:0'
    embedding_conf.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(summary_writer, config)

    save_file = os.path.join(save_dir, "model.ckpt")
    print("Saving session...")
    saver = tf.train.Saver()
    saver.save(sess, save_file)


def most_similar_intersection(inputs, word2vec):
    output = []
    similar = {}
    all_similar = []
    for word in inputs:
        sim = word2vec.wv.most_similar(word, topn=int(num_similar * (len(inputs)*2.5)))
        similar[word] = []
        for item in sim:
            w, n = item
            similar[word].append(w)
            if w not in all_similar:
                all_similar.append(w)
    subset = []
    for word in all_similar:
        is_subset = True
        for label, items in similar.iteritems():
            if word not in items:
                is_subset = False
        if is_subset == True:
            subset.append(word)
    if len(subset) > 0:
        output.append(inputs)
        output.append(subset)
    if len(output) > 0:
        return output
    else:
        return None

def most_similar(input_word, word2vec):
    sim = word2vec.wv.most_similar(input_word, topn=num_similar)
    output = []
    found = []
    for item in sim:
        w, n = item
        found.append(w)
    output = [input_word, found]
    t_sne_scatterplot(input_word, word2vec)
    return output

def nearest_similarity_cosmul(start1, end1, end2, word2vec):
    similarities = word2vec.wv.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

def test_word2vec(word2vec):
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    output = []
    test_items = []
    if 'test_words' in globals():
        print("Testing known words")
        test_items = test_words
    else:
        print("Using frequent words as test items")
        freq_file = os.path.join(save_dir, "cleaned_frequencies.json")
        frequencies = load_json(freq_file)
        if frequencies is not None:
            for item in frequencies[:50]:
                test_items.append(item[0])
    if len(test_items) > 0:
        for count, word in enumerate(test_items):
            if word in vocab:
                print("[" + str(count+1) + "] Testing: " + word)
                output.append(most_similar(word, word2vec))
            else:
                print("Word " + word + " not in vocab")
        filename = os.path.join(save_dir, "word2vec_test.json")
        save_json(output, filename)
    return output

def test_intersections(word2vec):
    output = []
    if 'test_groups' in globals():
        for count, group in enumerate(test_groups):
            print("[" + str(count+1) + "] Testing intersection: " + str(group))
            retval = most_similar_intersection(group, word2vec)
            if retval is not None:
                output.append(retval)
        filename = os.path.join(save_dir, "word2vec_test_groups.json")
        save_json(output, filename)
    return output

def show_cluster_locations(results, labels, x_coords, y_coords):
    big_plot_dir = os.path.join(save_dir, "big_plots")
    if not os.path.exists(big_plot_dir):
        os.makedirs(big_plot_dir)
    for item in results:
        name = item[0]
        print("Plotting big graph for " + name)
        filename = os.path.join(big_plot_dir, name + "_tsne.png")
        similar = item[1]
        in_set_x = []
        in_set_y = []
        out_set_x = []
        out_set_y = []
        name_x = 0
        name_y = 0
        for count, word in enumerate(labels):
            xc = x_coords[count]
            yc = y_coords[count]
            if word == name:
                name_x = xc
                name_y = yc
            elif word in similar:
                in_set_x.append(xc)
                in_set_y.append(yc)
            else:
                out_set_x.append(xc)
                out_set_y.append(yc)
        plt.figure(figsize=(16, 12), dpi=80)
        plt.scatter(name_x, name_y, s=400, marker="o", c="blue")
        plt.scatter(in_set_x, in_set_y, s=80, marker="o", c="red")
        plt.scatter(out_set_x, out_set_y, s=8, marker=".", c="black")
        if 'plot_lims' in globals():
            plt.xlim(plot_lims["xmin"], plot_lims["xmax"])
            plt.ylim(plot_lims["ymin"], plot_lims["ymax"])
        plt.savefig(filename)
        plt.close()


def plot_all(word2vec, clusters):
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    prog = vocab_len/50
    labels = []
    xkcd_colors = load_json("xkcd_color_names.json")
    arr = np.empty((0, dim0), dtype='f')
    labels = []
    vectors_file = os.path.join(save_dir, "vocab_vectors.npy")
    labels_file = os.path.join(save_dir, "labels.json")
    if os.path.exists(vectors_file) and os.path.exists(labels_file):
        print("Loading pre-saved vectors from disk")
        arr = load_bin(vectors_file)
        labels = load_json(labels_file)
    else:
        print("Creating an array of vectors for each word in the vocab")
        count = 0
        for cluster in clusters:
            for word in cluster:
                if count % prog == 0:
                    print_progress()
                w_vec = word2vec[word]
                labels.append(word)
                arr = np.append(arr, np.array([w_vec]), axis=0)
                count += 1
        save_bin(arr, vectors_file)
        save_json(labels, labels_file)
    x_c_filename = os.path.join(save_dir, "x_coords.npy")
    y_c_filename = os.path.join(save_dir, "y_coords.npy")
    x_coords = None
    y_coords = None
    if os.path.exists(x_c_filename) and os.path.exists(y_c_filename):
        print("Reading pre-calculated coords from disk")
        x_coords = load_bin(x_c_filename)
        y_coords = load_bin(y_c_filename)
    else:
        print("Computing T-SNE for array of length: " + str(len(arr)))
        tsne = TSNE(n_components=2, random_state=1, verbose=1)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)
        x_coords = Y[:, 0]
        y_coords = Y[:, 1]
        print("Saving calculated coords")
        save_bin(x_coords, x_c_filename)
        save_bin(y_coords, y_c_filename)
    index = 0
    plt.figure(figsize=(16, 12), dpi=80, facecolor='b', edgecolor='r')
    for count, cluster in enumerate(clusters):
        clen = len(cluster)
        if clen > 0:
            color = xkcd_colors[random.randint(0, len(xkcd_colors)-1)]
            print("Cluster: " + str(count) + " items: " + str(clen) + " color: " + color)
            xc = x_coords[index:index+clen]
            yc = y_coords[index:index+clen]
            plt.scatter(xc, yc, s=8, marker="o", c=color)
        else:
            print("Empty cluster")
            print cluster
        index += clen

    if 'plot_lims' in globals():
        plt.xlim(plot_lims["xmin"], plot_lims["xmax"])
        plt.ylim(plot_lims["ymin"], plot_lims["ymax"])
    plot_dir = os.path.join(save_dir, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    filename = os.path.join(save_dir, "all_vectors_tsne.png")
    plt.savefig(filename)
    plt.close()
    return labels, x_coords, y_coords

def t_sne_scatterplot(word, word2vec):
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    dim0 = word2vec.wv[vocab[0]].shape[0]

    arr = np.empty((0, dim0), dtype='f')
    w_labels = [word]
    nearby = word2vec.wv.similar_by_word(word, topn=num_similar)
    arr = np.append(arr, np.array([word2vec[word]]), axis=0)
    for score in nearby:
        w_vec = word2vec[score[0]]
        w_labels.append(score[0])
        arr = np.append(arr, np.array([w_vec]), axis=0)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.rc("font", size=16)
    plt.figure(figsize=(16, 12), dpi=80)
    plt.scatter(x_coords[0], y_coords[0], s=800, marker="o", color="blue")
    plt.scatter(x_coords[1:], y_coords[1:], s=200, marker="o", color="red")

    for label, x, y in zip(w_labels, x_coords, y_coords):
        plt.annotate(label.upper(), xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()-50, x_coords.max()+50)
    plt.ylim(y_coords.min()-50, y_coords.max()+50)
    plot_dir = os.path.join(save_dir, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    filename = os.path.join(plot_dir, word + "_tsne.png")
    plt.savefig(filename)
    plt.close()





if __name__ == '__main__':
    num_clusters = 30
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

#######################################
# Data preprocessing
#######################################
    full, cleaned = prepare_data()
    filename = os.path.join(save_dir, "cleaned_frequencies.json")
    cleaned_frequencies = get_word_frequencies(cleaned)
    save_json(cleaned_frequencies, filename)

#######################################
# Latent Dirichlet Analysis
#######################################
# LDA doesn't fucking work for shit
# Always returns n "topics" containing a set of the 6 most frequent words in the corpus
# If I had wanted that, I'd have fucking coded it in 3 lines myself
# Fucking bullshit
    #print
    #print("Performing LDA analysis")
    #LDA(cleaned)

#######################################
# word2vec
#######################################
    print
    print("Instantiating word2vec model")
    word2vec = get_word2vec(cleaned)
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    print("word2vec vocab contains " + str(vocab_len) + " items.")
    dim0 = word2vec.wv[vocab[0]].shape[0]
    print("word2vec items have " + str(dim0) + " features.")
    create_embeddings(word2vec)

    print("Running intersection tests")
    intersections = test_intersections(word2vec)

#######################################
# k-means clustering
#######################################
    print
    clusters = None
    filename = os.path.join(save_dir, "clusters.json")
    if os.path.exists(filename):
        clusters = load_json(filename)
    else:
        print("Clustering (k=" + str(num_clusters)+")")
        clusters = k_means_cluster(word2vec, num_clusters)

    print("Plotting full graph")
    labels, x_coords, y_coords = plot_all(word2vec, clusters)

    print("Running tests")
    results = test_word2vec(word2vec)

    print("Plotting test results on full cluster diagrams")
    show_cluster_locations(results, labels, x_coords, y_coords)

    print
    print("Done")
