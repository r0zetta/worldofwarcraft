import gensim.models.word2vec as w2v
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

def most_similar(input_word):
    if input_word not in word_vectors.wv.vocab:
        print(input_word + " was not in the vocabulary.")
        return
    print("Most similar words to: " + input_word)
    sim = word_vectors.wv.most_similar(input_word)
    found = []
    for item in sim:
        w, n = item
        found.append(w)
    print(", ".join(found))
    print("")

def nearest_similarity_cosmul(start1, end1, end2):
    similarities = word_vectors.wv.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2

if __name__ == '__main__':
    print("Loading pretrained model...")
    word_vectors = w2v.Word2Vec.load("save/word_vectors.w2v")

    all_word_vectors_matrix = word_vectors.wv.syn0
    print("Number of word vectors: " + str(len(all_word_vectors_matrix)))

    most_similar("Mage")
    most_similar("Priest")
    most_similar("Paladin")
    most_similar("Monk")
    most_similar("Warrior")
    most_similar("Hunter")
    most_similar("Warlock")
    most_similar("nerf")
    most_similar("OP")

    print("")
    nearest_similarity_cosmul("Mage", "DPS", "Priest")
    print("")

