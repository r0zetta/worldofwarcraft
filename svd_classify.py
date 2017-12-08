from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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

if __name__ == '__main__':
    sentences = []
    input_file = "data/data.json"
    raw_data = load_data(input_file)
    sentences = split_into_sentences(raw_data)
    doc_complete = sentences

    print("Vectorizing data")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(doc_complete)

    print("Performing SVD analysis")
    lsa = TruncatedSVD(n_components=300, n_iter=100)
    lsa.fit(X)

    terms = vectorizer.get_feature_names()
    for i,comp in enumerate(lsa.components_):
        termsInComp = zip(terms,comp)
        sortedterms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:25]
        outstring = ""
        count = 0
        for term in sortedterms:
            if count == 0:
                outstring += upperfirst(term[0]) + " "
            else:
                outstring += term[0] + " "
            count += 1
        outstring = outstring.strip()
        outstring += "."
        print outstring
