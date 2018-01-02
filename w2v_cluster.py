import gensim.models.word2vec as w2v
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from text_handler import split_line_into_words
import re
import logging
import multiprocessing
import os
import nltk
import json
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
save_dir = "w2v_cluster"

def print_progress():
    sys.stdout.write("#")
    sys.stdout.flush()

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
    count = 0
    for s in sentences:
        if len(s) > 0:
            tokens, lost = split_line_into_words(s)
            if len(tokens) > 0:
                ret.append(tokens)
        if count % prog_at == 0:
            print_progress()
        count += 1
    return ret

def clean_sentences(tokens, stopwords, stemmer):
    ret = []
    max_s = len(tokens)
    prog_at = max_s / 50
    count = 0
    for sentence in tokens:
        if count % prog_at == 0:
            print_progress()
        count += 1
        cleaned = []
        for token in sentence:
            if re.search("^\w+$", token):
                token = token.lower()
                stw = False
                if stopwords is not None:
                    for s in stopwords:
                        if token == s:
                            #print("Removed stopword " + token)
                            stw = True
                            token = None
                            break
                if stw == False:
                    stem = stemmer.stem(token)
                    if stem is not None:
                        #print("Stemmed " + token + " to " + stem)
                        token = stem
                if token is not None:
                    cleaned.append(token)
        ret.append(cleaned)
    return ret

def load_raw_data(input_files):
    ret = ""
    for x in input_files:
        if os.path.exists(x):
            raw = load_json(x)
            ret += "\n".join(raw)
    return ret

if __name__ == '__main__':
    num_workers = multiprocessing.cpu_count()
    num_features = 300
    epoch_count = 10
    num_clusters = 10

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cleaned_file = os.path.join(save_dir, "cleaned.json")
    cleaned = load_json(cleaned_file)
    if cleaned is None:
        tokens_file = os.path.join(save_dir, "tokens.json")
        tokens = load_json(tokens_file)
        if tokens is None:
            input_files = ["battle_net_data/data.json", "mmo_champion_data/data.json"]
            print("Loading raw data")
            raw_data = load_raw_data(input_files)

            print("Splitting data into sentences")
            raw_sentences = split_into_sentences(raw_data)
            print("Tokenizing sentences")
            tokens = tokenize_sentences(raw_sentences)
            save_json(tokens, tokens_file)

        sentence_count = len(tokens)
        print("Number of sentences: " + str(sentence_count))
        token_count = sum([len(sentence) for sentence in tokens])
        print("The corpus contains " + str(token_count) + " tokens")
        print("Cleaning and stemming data")

        stopwords_file = "data/stopwords-iso.json"
        stopwords = load_json(stopwords_file)
        stopwords_en = None
        if stopwords is not None:
            stopwords_en = stopwords["en"]
        stemmer = SnowballStemmer("english")
        cleaned = clean_sentences(tokens, stopwords_en, stemmer)
        save_json(cleaned, cleaned_file)

    sentence_count = len(cleaned)
    print("Number of sentences: " + str(sentence_count))
    token_count = sum([len(sentence) for sentence in cleaned])
    print("The corpus contains " + str(token_count) + " tokens")

    sentences = cleaned

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
                                min_count=0,
                                window=7,
                                sample=0)

        print("Building vocab...")
        word2vec.build_vocab(sentences)
        print("Word2Vec vocabulary length:", len(word2vec.wv.vocab))
        print("Training...")
        word2vec.train(sentences, total_examples=sentence_count, epochs=epoch_count)
        print("Saving model...")
        word2vec.save(w2v_file)

    print("Clustering (k=" + str(num_clusters)+")")
    word_vectors = word2vec.wv.syn0
    kmeans_clustering = KMeans(n_clusters = num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)
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
    print("Done")

