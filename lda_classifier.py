from sklearn.feature_extraction.text import TfidfVectorizer
from text_handler import load_and_tokenize, split_line_into_words, split_input_into_sentences
from gensim import corpora, models, similarities 
from nltk.stem.snowball import SnowballStemmer
import json
import re
import os

def filter_tokens(tokens):
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def stem_tokens(tokens):
    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def filter_and_stem(tokens):
    filtered = filter_tokens(tokens)
    stems = stem_tokens(filtered)
    return stems

if __name__ == '__main__':
    input_file = "data/data.json"
    tokens_file = "data/tokens.json"
    sentences_file = "data/sentences.json"
    split_mode = "words"
    raw_data = []
    tokens = []
    sentences = []
    print("Loading and tokenizing input data")
    with open(input_file, "r") as f:
        raw_data = json.load(f)
    if not os.path.exists(tokens_file):
        tokens = load_and_tokenize(input_file, split_mode)
        with open(tokens_file, "w") as f:
            json.dump(tokens, f, indent=4)
    else:
        with open(tokens_file, "r") as f:
            tokens = json.load(f)
    if not os.path.exists(sentences_file):
        sentences = split_input_into_sentences(raw_data)
        with open(sentences_file, "w") as f:
            json.dump(sentences, f, indent=4)
    else:
        with open(sentences_file, "r") as f:
            sentences = json.load(f)
    print("Filtering and stemming")
    #texts = filter_and_stem(tokens)

    print("Creating dictionary")
    dictionary = corpora.Dictionary(sentences)
    #dictionary.filter_extremes(no_below=1, no_above=0.8)

    print("Creating corpus")
    corpus = [dictionary.doc2bow(tokens)]

    print("Building model")
    lda = models.LdaModel(corpus, num_topics=50, 
                          id2word=dictionary, 
                          update_every=5, 
                          random_state=1,
                          chunksize=10000, 
                          passes=100)

    topics_matrix = lda.show_topics(formatted=False, num_words=20)
    topic_words = topics_matrix[:,:,1]
    for w in topic_words:
        print([str(word) for word in w])
        print

