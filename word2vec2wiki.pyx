import cython
cimport cython
import time
import numpy as np
cimport numpy as np
from cpython cimport array
import pickle


from corpus import process_corpus, create_sample_table_and_undersampling_table
from training import update_gradients_batch
cimport training

#All parameters related to training are set up here

cdef bint corpus_processed = 1
cdef bint sample_table_created = 1
cdef double start_step = 0.025
cdef double step = 0.025
cdef int epochs = 15
cdef int window_size_param = 10
cdef int num_neg_samples = 10
cdef int num_features = 100
cdef int cut_off = 25
cdef double sample_constant = 1e-3
cdef int batch_size = 100000000
cdef int num_batches
cdef int last_batch_size
cdef int batch_len

cdef int voc_size = 0
cdef int corpus_size = 0
cdef array.array next_random = array.array('Q', [1])


cdef dict word2ind = dict()
cdef dict ind2word = dict()
cdef dict ind2count = dict()

cdef array.array word2prob = array.array('f')
cdef array.array rand2sample = array.array('i')
cdef array.array undersampling_table = array.array('f')
cdef list sentences

#Filepath to a text file composed of newline-separated sentences should be provided here

file_path = r"corpus_fp"


#Processes a corpus into pickled files of word indexes, creating all needed dictionaries along the way
#The translation into word indexes is a poor way of doing things, for the simple reason that it doubles the storage requirement at little to no improvement in speed
#For larger corpuses this is infeasible

if not corpus_processed:
    corpus_size, voc_size, num_batches, last_batch_size = process_corpus(file_path, word2ind, ind2word, ind2count, cut_off, 100000000, True)
else:
    with open("pickled/word2ind.pickle", "rb") as w2i:
        word2ind = pickle.load(w2i)
    with open("pickled/ind2word.pickle", "rb") as i2w:
        ind2word = pickle.load(i2w)
    with open("pickled/ind2count.pickle", "rb") as i2c:
        ind2count = pickle.load(i2c)           
    with open("pickled/corpus_info.pickle", "rb") as ci:
        info = pickle.load(ci)
        corpus_size = info['corpus_size']
        voc_size = info['voc_size']
        num_batches = info['num_batches']
        last_batch_size = info['last_batch_size']


#Creates the sample table and undersampling table as used in negative sampling

if not sample_table_created:
    rand2sample, undersampling_table = create_sample_table_and_undersampling_table(ind2count, corpus_size, pickle_tables = True)
else:
    with open("pickled/rand2sample.pickle", "rb") as r2s:
        rand2sample = pickle.load(r2s)
    with open("pickled/undersampling_table.pickle", "rb") as ut:
        undersampling_table = pickle.load(ut)

cdef int sample_table_size = len(rand2sample)

print("started training")

start = time.time()

cdef double loss = 0

center_word_vectors = (np.random.rand(voc_size, num_features).astype(np.float32)-0.5)/num_features
outside_word_vectors = np.zeros((voc_size,num_features)).astype(np.float32)

for epoch in range(epochs):
    print("epoch: " + str(epoch))
    for batch in range(num_batches):
        print("batch: " + str(batch))
        with open("pickled/pickled_sentences_" + str(batch) + ".pickle", 'rb') as pickled_sentences:
            sentences = pickle.load(pickled_sentences)

        batch_len = batch_size if batch != num_batches-1 else last_batch_size
        loss = update_gradients_batch(center_word_vectors, outside_word_vectors, sentences, batch_len, num_features, rand2sample, sample_table_size, num_neg_samples, window_size_param, step, undersampling_table, next_random, 1)

        step = start_step*(epochs*num_batches - epoch*num_batches - batch)/(epochs*num_batches)
        if step < 0.0001:
            step = 0.0001

        print("loss is " + str(loss/batch_len))
        print("step is " + str(step))


end = time.time()

print("time elapsed: " + str(end - start))

word_vectors = center_word_vectors

for i in range(voc_size):
    word_vectors[i] /= np.linalg.norm(word_vectors[i])

with open("word2vec.pickle", 'wb') as word2vec:
    pickle.dump(word_vectors, word2vec)
