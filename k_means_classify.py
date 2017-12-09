from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os
import nltk
import json

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

def predict(sentence):
    Y = vectorizer.transform([sentence])
    prediction = model.predict(Y)
    print("Prediction for \"" + sentence + "\": " + str(prediction))
    return prediction

def predict_val(sentence):
    Y = vectorizer.transform([sentence])
    prediction = model.predict(Y)
    return prediction

if __name__ == '__main__':
    sentences = []
    input_file = "data/data.json"
    json_data = load_data(input_file)
    raw_data = "\n".join(json_data)
    documents = split_into_sentences(raw_data)

    print("Vectorizing")
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)

    print("Clustering")
    num_k = 200
    model = KMeans(n_clusters=num_k, init='k-means++', max_iter=200, n_init=1)
    model.fit(X)

    print("Top terms per cluster:")
    term_labels = {}
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :25]:
            print(' %s' % terms[ind]),
            term_labels[i] = terms[ind]
        print
    with open("save/term_labels.json", "w") as f:
        json.dump(term_labels, f, indent=4)

    print("\n")
    print("Prediction")
    predict("y u nerf shadowpriest?")
    predict("Warlocks are OP.")
    predict("Good job Blizz.")
    predict("Tank balance is bad.")

    print("Running prediction on all input text strings.")
    predictions = {}
    counts = {}
    c = 0
    for s in json_data:
        category = predict_val(s)
        if c % 100 == 0:
            print("Processed " + str(c) + " samples.")
        c += 1
        if category not in predictions:
            predictions[category] = [s]
        else:
            predictions[category].append(s)
        if category not in counts:
            counts[category] = 1
        else:
            counts[category] += 1
    with open("save/predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
    with open("save/counts.json", "w") as f:
        json.dump(predictions, f, indent=4)

