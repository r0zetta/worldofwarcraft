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
    save_dir = "lda/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    input_file = "data/data.json"
    tokens_file = os.path.join(save_dir, "tokens.json")
    sentences_file = os.path.join(save_dir, "sentences.json")
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

    model_save_name = os.path.join(save_dir, "lda_model.sav")
    lda = None
    if os.path.exists(model_save_name):
        print("Loading model from disk")
        lda = models.LdaModel.load(model_save_name)
    else:
        print("Creating corpus")
        corpus = [dictionary.doc2bow(tokens)]
        print("Building model")
        lda = models.LdaModel(corpus, num_topics=50, 
                              id2word=dictionary, 
                              update_every=5, 
                              random_state=1,
                              chunksize=10000, 
                              passes=100)
        lda.save(model_save_name)

    topics_matrix = lda.show_topics(formatted=False, num_words=20)
    topics_save_name = os.path.join(save_dir, "topics_matrix.json")
    with open(topics_save_name, "w") as f:
        json.dump(topics_matrix, f, indent=4)
    sentence_map = {}
    for s in sentences:
        chunk = dictionary.doc2bow(s)
        print chunk
        sentence_map[s] = lda.get_document_topics(chunk)
    sentence_map_save = os.path.join(save_dir, "sentence_map.json")
    with open(sentence_map_save, "w") as f:
        json.dump(topics_matrix, f, indent=4)
