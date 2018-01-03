from gensim import corpora, models, similarities 
import gensim.models.word2vec as w2v
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from text_handler import split_line_into_words
import numpy as np
import matplotlib.pyplot as plt
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
        cleaned1 = clean_sentences(tokens, stopwords_en, stemmer)
        cleaned = []
        for sent in cleaned1:
            if len(sent) > 0:
                cleaned.append(sent)
            else:
                print("Found empty sentence. WTF??!?!?!")
        save_json(cleaned, cleaned_file)
    return cleaned




def get_word2vec(sentences):
    num_workers = multiprocessing.cpu_count()
    num_features = 500
    epoch_count = 100
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
                                sample=0.0001)

        print("Building vocab...")
        word2vec.build_vocab(sentences)
        print("Word2Vec vocabulary length:", len(word2vec.wv.vocab))
        print("Training...")
        word2vec.train(sentences, total_examples=sentence_count, epochs=epoch_count)
        print("Saving model...")
        word2vec.save(w2v_file)
    return word2vec

def most_similar(input_word, word2vec):
    input_word = input_word.lower
    if input_word not in word2vec.wv.vocab:
        print(input_word + " was not in the vocabulary.")
        return
    sim = word2vec.wv.most_similar(input_word)
    found = []
    for item in sim:
        w, n = item
        found.append(w)
    print(input_word + ": " + ", ".join(found))
    t_sne_scatterplot(word2vec, input_word)

def nearest_similarity_cosmul(start1, end1, end2, word2vec):
    similarities = word2vec.wv.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

def t_sne_scatterplot(word2vec, word):
    vocab = word_vectors.wv.vocab
    vocab_len = len(vocab)
    print("word2vec vocab contains " + str(vocab_len) + " items.")
    dim0 = word2vec.wv[vocab[0]].shape[0]
    print("word2vec items have " + str(dim0) + " features.")

    word = word.lower()
    if word not in vocab:
        print("Queried word was not in vocab")
        return

    arr = np.empty((0, dim0), dtype='f')
    w_labels = [word]
    nearby = word2vec.similar_by_word(word)
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
    plt.scatter(x_coords, y_coords)


    for label, x, y in zip(w_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    filename = os.path.join(save_dir, word + "_tsne.png")
    plt.savefig(filename)

def test_word2vec(word2vec):
    print
    most_similar("Mage", word2vec)
    most_similar("Elf", word2vec)
    most_similar("Void", word2vec)
    print
    most_similar("PVP", word2vec)
    most_similar("gank", word2vec)
    print
    most_similar("PVE", word2vec)
    most_similar("Raid", word2vec)
    most_similar("raiding", word2vec)
    most_similar("Nighthold", word2vec)
    most_similar("Varimathras", word2vec)
    print
    most_similar("Alliance", word2vec)
    most_similar("Horde", word2vec)
    most_similar("evil", word2vec)
    most_similar("good", word2vec)
    print


def LDA(tokens):
    sentences = tokens
    print("Creating dictionary")
    dictionary = corpora.Dictionary(sentences)

    model_save_name = os.path.join(save_dir, "lda_model.sav")
    lda = None
    if os.path.exists(model_save_name):
        print("Loading model from disk")
        lda = models.LdaModel.load(model_save_name)
    else:
        print("Creating corpus")
        corpus = [dictionary.doc2bow(sentences)]
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
    sentence_map_save = save_dir + "sentence_map.json"
    with open(sentence_map_save, "w") as f:
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



if __name__ == '__main__':
    num_clusters = 30
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

#######################################
# Data preprocessing
#######################################
    sentences = prepare_data()
    sentence_count = len(sentences)
    print("Number of sentences: " + str(sentence_count))
    token_count = sum([len(sentence) for sentence in sentences])
    print("The corpus contains " + str(token_count) + " tokens")

#######################################
# Latent Dirichlet Analysis
#######################################
    print("Performing LDA analysis")
    LDA(sentences)

#######################################
# word2vec
#######################################
    print("Instantiating word2vec model")
    word2vec = get_word2vec(sentences)
    vocab = word_vectors.wv.vocab
    vocab_len = len(vocab)
    print("word2vec vocab contains " + str(vocab_len) + " items.")
    dim0 = word2vec.wv[vocab[0]].shape[0]
    print("word2vec items have " + str(dim0) + " features.")
    test_word2vec(word2vec)




#######################################
# k-means clustering
#######################################
    print("Clustering (k=" + str(num_clusters)+")")
    k_means_cluster(word2vec, num_clusters)

