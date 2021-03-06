from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import argparse
import pickle
import os
import sys
import io
import json

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

def predict_val(sentence, vectorizer, model):
    Y = vectorizer.transform([sentence])
    prediction = model.predict(Y)
    return int(prediction[0])

# XXX implement
def preprocess(data_set):
    return data_set

def vectorize(documents):
    print("Vectorizing with tf-idf")
    vectorizer = TfidfVectorizer(lowercase=False)
    vectors = vectorizer.fit_transform(documents)
    vocab = vectorizer.vocabulary_
    words = list(vocab.keys())
    return vectors, vectorizer

def cluster(prefix, vectors, vectorizer, k):
    print("k-means clustering with k=" + str(k))
    print("Prefix: " + prefix)
    model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, random_state=random_seed, verbose=1)
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
    for i in range(k):
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
        print("Running predictions.")
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

def analyze(prefix, data_set, k):
    data_set = preprocess(data_set)
    X, v = vectorize(data_set)
    m = cluster(prefix, X, v, k)
    p, c = run_predictions(prefix, v, m, data_set)
    return p, c

def dump_text(prefix, data_set):
    filename = cluster_dir + prefix + "cluster.txt"
    with io.open(filename, "w", encoding="utf-8") as f:
        for d in data_set:
            f.write(d + u"\n")

def expand_clusters(prefix, counts, predictions, num_samples, depth, k):
    print("Expand clusters called with prefix: " + prefix + ", num_samples=" + str(num_samples) + " k=" + str(k))
    depth += 1
    c_vals = np.array(list(counts.values()))
    c_std = int(np.std(c_vals))
    c_mean = int(np.mean(c_vals))
    print c_vals
    print c_std
    print c_mean

    for index, value in counts.iteritems():
        new_prefix = prefix + str(index) + "_"
        data_set = predictions[index]
        print("index: " + str(index) + " value:" + str(value) + " k: " + str(k))
        dump_text(new_prefix, data_set)
        if c_std > c_mean:
            if k > 2 and value > k:
                if value > int(c_mean * 1.5) and value > min_samples_to_cluster:
                    print("Analyzing...")
                    new_k = int(k / cluster_decay_rate)
                    p, c = analyze(new_prefix, data_set, new_k)
                    if depth < max_depth:
                        expand_clusters(new_prefix, c, p, value, depth, new_k)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=30,
                       help='Number of centroids')
    parser.add_argument('--max_depth', type=int, default=3,
                       help='maximum recursion depth')
    parser.add_argument('--random_seed', type=int, default=1,
                       help='random seed')
    parser.add_argument('--min_to_cluster', type=int, default=100,
                       help='minimum number of samples to force a new cluster')
    parser.add_argument('--cluster_decay_rate', type=float, default=3.0,
                       help='rate at which clusters shrink per pass')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    num_k = args.k
    cluster_decay_rate = args.cluster_decay_rate
    max_depth = args.max_depth
    min_samples_to_cluster = args.min_to_cluster
    random_seed = args.random_seed
    save_dir = "k_means_ " + str(num_k)+ "/"
    cluster_dir = save_dir + "clusters/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    use_dim_reduction = False
    json_data = load_data("k_means_data/data.json")
    documents = json_data

# Perhaps run this in a loop, optimizing num_k
    num_samples = len(documents)
    print("Read in " + str(num_samples) + " lines.")
    predictions, counts = analyze("", documents, num_k)
    expand_clusters("", counts, predictions, num_samples, 0, num_k)

