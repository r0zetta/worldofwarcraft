import gensim.models.word2vec as w2v
from text_handler import split_line_into_words
import logging
import multiprocessing
import os
import nltk
import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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

def tokenize_sentences(sentences):
    ret = []
    print("Tokenizing sentences.")
    for s in sentences:
        if len(s) > 0:
            tokens, lost = split_line_into_words(s)
            if len(tokens) > 0:
                ret.append(tokens)
    return ret

if __name__ == '__main__':
    input_file = "data/data.json"
    num_features = 500
    min_word_count = 3
    num_workers = multiprocessing.cpu_count()
    context_size = 7
    downsampling = 1e-3
    seed = 1
    epoch_count = 1000

    raw_data = load_data(input_file)
    raw_sentences = split_into_sentences(raw_data)
    sentences = tokenize_sentences(raw_sentences)

    sentence_count = len(sentences)
    print("Number of sentences: " + str(sentence_count))
    token_count = sum([len(sentence) for sentence in sentences])
    print("The corpus contains " + str(token_count) + " tokens")

    word_vectors = w2v.Word2Vec(sg=1,
                                seed=seed,
                                workers=num_workers,
                                size=num_features,
                                min_count=min_word_count,
                                window=context_size,
                                sample=downsampling)

    print("Building vocab...")
    word_vectors.build_vocab(sentences)

    print("Word2Vec vocabulary length:", len(word_vectors.wv.vocab))

    print("Training...")
    word_vectors.train(sentences, total_examples=sentence_count, epochs=epoch_count)

    if not os.path.exists("save"):
        os.makedirs("save")
    print("Saving model...")
    word_vectors.save(os.path.join("save/word_vectors.w2v"))

    print("Done")
