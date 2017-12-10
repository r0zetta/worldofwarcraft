import os
import tensorflow as tf
import numpy as np
import gensim.models.word2vec as w2v
from tensorflow.contrib.tensorboard.plugins import projector
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

save_dir = "embeddings/"

if __name__ == '__main__':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(save_dir + "word_vectors.w2v"):
        print("Loading model...")
        word2vec = w2v.Word2Vec.load(save_dir + "word_vectors.w2v")
    else:
        assert False, "No saved model"

    all_word_vectors_matrix = word2vec.wv.syn0
    num_words = len(all_word_vectors_matrix)
    print("Number of word vectors: " + str(num_words))

    vocab = word2vec.wv.vocab
    print("Vocab length: " + str(len(vocab)))
    dim = word2vec.wv["."].shape[0]
    print("Tensor dimensions: " + str(dim))

    print("Creating embeddings list...")
    embedding = np.empty((num_words, dim), dtype=np.float32)
    i = 0
    for word, obj in vocab.iteritems():
        embedding[i] = word2vec.wv[word]
        i += 1

    print("Running tensorflow session...")
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    X = tf.Variable([0.0], name='embedding')
    place = tf.placeholder(tf.float32, shape=embedding.shape)
    set_x = tf.assign(X, place, validate_shape=False)
    sess.run(tf.global_variables_initializer())
    sess.run(set_x, feed_dict={place: embedding})

    print("Writing word vectors...")
    with open(save_dir + "metadata.tsv", "w") as f:
        for word, obj in vocab.iteritems():
            f.write(word + '\n')

    summary_writer = tf.summary.FileWriter(save_dir, sess.graph)
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = 'embedding:0'
    embedding_conf.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(summary_writer, config)

    print("Saving session...")
    saver = tf.train.Saver()
    saver.save(sess, save_dir + "model.ckpt")
    print("To show embeddings, run:")
    print("tensorboard --logdir=" + save_dir)
