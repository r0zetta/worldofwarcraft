from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import numpy as np
import pickle
import os
import sys
import io
import nltk
import json
import random

stopwords = ""

def print_progress():
    sys.stdout.write("#")
    sys.stdout.flush()

def upperfirst(x):
    return x[0].upper() + x[1:]

def load_data(input_file):
    raw_data = []
    print("Loading raw data")
    if os.path.exists(input_file):
        with open(input_file, "r") as f:
            raw_data = json.load(f)
    return raw_data

def split_into_sentences(raw_data):
    nltk.download("punkt")
    nltk.download("stopwords")
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    print("Splitting raw data into sentences.")
    raw_sentences = tokenizer.tokenize(raw_data)
    num_raw_sentences = len(raw_sentences)
    print("Raw sentence count: " + str(num_raw_sentences))
    return raw_sentences

def predict_val(sentence, vectorizer, model):
    Y = vectorizer.transform([sentence])
    Y = dim_reduction(Y)
    prediction = model.predict(Y)
    return int(prediction[0])

def get_vocab(documents):
    print("Getting vocab.")
    vectorizer = CountVectorizer(lowercase=False)
    vectorizer.fit(documents)
    vocab = vectorizer.vocabulary_
    words = list(vocab.keys())
    return words

def vectorize(documents):
    print("Vectorizing with tf-idf")
    if stopwords != "":
        vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=False)
    else:
        vectorizer = TfidfVectorizer(lowercase=False)
    vectors = vectorizer.fit_transform(documents)
    vocab = vectorizer.vocabulary_
    words = list(vocab.keys())
    return vectors, vectorizer

def dim_reduction(X):
    if use_dim_reduction == False:
        return X
    svd = TruncatedSVD(n_components=num_c, n_iter=100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    return X

def cluster(prefix, vectors, vectorizer, num_k):
    print("k-means clustering with k=" + str(num_k))
    print("Prefix: " + prefix)
    model = KMeans(n_clusters=num_k, init='k-means++', max_iter=100, n_init=1, verbose=1)
    model_save_path = save_dir + prefix + "k_means_model.sav"
    if os.path.exists(model_save_path):
        print("Loading model from disk.")
        model = pickle.load(open(model_save_path, "rb"))
    else:
        print("Clustering")
        model.fit(vectors)
        print("Saving model to: " +  model_save_path)
        pickle.dump(model, open(model_save_path, "wb"))
    term_labels = {}
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    terms_save_path = save_dir + prefix + "term_labels.json"
    for i in range(num_k):
        term_labels[i] = []
        for ind in order_centroids[i, :25]:
            term_labels[i].append(terms[ind])
    print("Saving terms labels to: " + terms_save_path)
    if not os.path.exists(terms_save_path):
        with open(terms_save_path, "w") as f:
            json.dump(term_labels, f, indent=4)
    return model

def run_predictions(prefix, vectorizer, model, data_set):
    predictions = {}
    counts = {}
    print("Prefix: " + prefix)
    print("Running predictions")
    predictions_file = save_dir + prefix + "predictions.json"
    counts_file = save_dir + prefix + "counts.json"
    if os.path.exists(predictions_file):
        with open(predictions_file, "r") as f:
            print("Loading predictions from: " + predictions_file)
            predictions = json.load(f)
    if os.path.exists(counts_file):
        with open(counts_file, "r") as f:
            print("Loading counts from: " + counts_file)
            counts = json.load(f)
    if len(predictions) < 1 and len(counts) < 1:
        print("Running prediction on data_set strings.")
        c = 0
        for s in data_set:
            category = predict_val(s, vectorizer, model)
            if c % 100 == 0:
                print_progress()
            c += 1
            if category not in predictions:
                predictions[category] = [s]
            else:
                predictions[category].append(s)
            if category not in counts:
                counts[category] = 1
            else:
                counts[category] += 1
        with open(predictions_file, "w") as f:
            print("Saving predictions to: " + predictions_file)
            json.dump(predictions, f, indent=4)
        with open(counts_file, "w") as f:
            print("Saving counts to: " + counts_file)
            json.dump(counts, f, indent=4)
    return predictions, counts

def analyze(prefix, data_set, num_k):
    X, v = vectorize(data_set)
    X = dim_reduction(X)
    m = cluster(prefix, X, v, num_k)
    p, c = run_predictions(prefix, v, m, data_set)
    return p, c

def dump_text(prefix, data_set):
    filename = cluster_dir + prefix + "cluster.txt"
    with io.open(filename, "w", encoding="utf-8") as f:
        for d in data_set:
            f.write(d + u"\n")

def expand_clusters(prefix, counts, predictions, num_samples, depth):
    print("Expand clusters called with prefix: " + prefix + ", num_samples=" + str(num_samples))
    depth += 1
    c_vals = np.array(list(counts.values()))
    c_std = np.std(c_vals)
    c_mean = np.mean(c_vals)
    print c_vals
    print c_std
    print c_mean

    for index, value in counts.iteritems():
        new_prefix = prefix + str(index) + "_"
        data_set = predictions[index]
        n_samples = len(data_set)
        new_k = num_k / int(cluster_decay_rate ** depth)
        print("index: " + str(index) + " n_samples: " + str(n_samples) + "new_k: " + str(new_k))
        dump_text(new_prefix, data_set)
        if c_std > 200 and value > c_mean:
            if new_k > 2 and new_k > n_samples and n_samples > min_samples_to_cluster:
                p, c = analyze(new_prefix, data_set, new_k)
                if depth <= max_depth:
                    expand_clusters(new_prefix, c, p, n_samples, depth)

if __name__ == '__main__':
    num_c = 300
    num_k = 30
    cluster_split_threshold = 0.05
    cluster_decay_rate = 2
    max_depth = 3
    min_samples_to_cluster = 100
    random.seed(1)
    save_dir = "k_means_ " + str(num_k)+ "/"
    cluster_dir = save_dir + "clusters/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    use_dim_reduction = False
    split_into_sentences = False
    json_data = load_data("data/data.json")
    documents = []
    if split_into_sentences == True:
        documents = split_into_sentences("\n".join(json_data))
    else:
        documents = json_data

# Perhaps run this in a loop, optimizing num_k
    num_samples = len(documents)
    print("Read in " + str(num_samples) + " lines.")
    predictions, counts = analyze("", documents, num_k)
    expand_clusters("", counts, predictions, num_samples, 0)

