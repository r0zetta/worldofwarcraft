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
    return "\n".join(raw_data)

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


if __name__ == '__main__':
    sentences = []
    input_file = "data/data.json"
    raw_data = load_data(input_file)
    documents = split_into_sentences(raw_data)

    print("Vectorizing")
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)

    print("Clustering")
    num_k = 200
    model = KMeans(n_clusters=num_k, init='k-means++', max_iter=200, n_init=1)
    model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :25]:
            print(' %s' % terms[ind]),
        print

    print("\n")
    print("Prediction")
    predict("y u nerf shadowpriest?")
    predict("Warlocks are OP.")
    predict("Good job Blizz.")
    predict("Tank balance is bad.")
