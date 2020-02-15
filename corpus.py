import pickle
import array
import string


def process_corpus(fp, word2ind, ind2word, ind2count, cut_off = 25, batch_size = 100000000, pickle_metadata = False):



    corpus_size = 0
    voc_size = 0
    last_batch_size = 0

    _word2ind = dict()
    _ind2word = dict()
    _ind2count = dict()


    with open(fp, 'r', encoding = "ansi") as corpus_file:
        #data = []
        for line in corpus_file:
            if line != '' and len(line) > 5:
                line = line.translate(str.maketrans('','',string.punctuation))
                tokens = line.split()
                filtered_tokens = [word.lower() for word in tokens] 
                hashed_sentence = []

                for word in filtered_tokens:

                    if word not in _word2ind and word != '':
                        _word2ind[word] = voc_size
                        _ind2word[voc_size] = word
                        voc_size +=1
                    if _word2ind[word] in _ind2count:
                        _ind2count[_word2ind[word]] += 1
                    else:
                        _ind2count[_word2ind[word]] = 1

    voc_size = 0
    for ind in _ind2count:
        if _ind2count[ind] > cut_off:
            corpus_size += _ind2count[ind]
            word2ind[_ind2word[ind]] = voc_size
            ind2word[voc_size] = _ind2word[ind]
            ind2count[voc_size] = _ind2count[ind]
            voc_size += 1

    print(voc_size)


    num_batches = 1
    pickled_sentences = open("pickled/pickled_sentences_0.pickle", 'wb')
    with open(fp, 'r', encoding = "ansi") as corpus_file:
        data = []
        for line in corpus_file:
            if line != '' and len(line) > 5:
                line = line.translate(str.maketrans('','',string.punctuation))
                tokens = line.split()
                hashed_sentence = array.array('i',[word2ind[word.lower()] for word in tokens if word.lower() in word2ind])      
                last_batch_size += len(hashed_sentence)
                data.append(hashed_sentence)

            if last_batch_size > batch_size:
                pickle.dump(data, pickled_sentences)
                pickled_sentences.close()
                pickled_sentences = open("pickled/pickled_sentences_" + str(num_batches) + ".pickle", 'wb')
                num_batches += 1
                data = []
                last_batch_size = 0
    pickle.dump(data, pickled_sentences) 
    pickled_sentences.close()



    if pickle_metadata:
        with open("pickled/word2ind.pickle", "wb") as w2i:
            pickle.dump(word2ind,w2i)
        with open("pickled/ind2word.pickle", "wb") as i2w:
            pickle.dump(ind2word,i2w)
        with open("pickled/ind2count.pickle", "wb") as i2c:
            pickle.dump(ind2count,i2c)              
        with open("pickled/corpus_info.pickle", "wb") as ci:
            pickle.dump({'corpus_size':corpus_size,'voc_size':voc_size, 'cut-off': cut_off, 'num_batches':num_batches,'last_batch_size':last_batch_size}, ci) 

    return corpus_size, voc_size, num_batches, last_batch_size


def create_sample_table_and_undersampling_table(ind2count, corpus_size, size = 100000000, power = 0.75, sample_constant = 1e-3, pickle_tables = False):
    powered_sum = 0
    voc_size = len(ind2count)

    word2prob = array.array('f')
    rand2sample = array.array('i')
    undersampling_table = array.array('f')

    for i in range(len(ind2count)):
        word2prob.append(ind2count[i] ** power)
        powered_sum += word2prob[i]
        undersampling_table.append(((ind2count[i]/(sample_constant * corpus_size))**0.5 + 1) * (sample_constant * corpus_size) / ind2count[i])

    for i in range(voc_size):
        word2prob[i] /= powered_sum

    for i in range(voc_size):
        for j in range(int(word2prob[i] * size)):
            rand2sample.append(i)

    if pickle:
        with open("pickled/rand2sample.pickle", "wb") as r2s:
            pickle.dump(rand2sample,r2s)
        with open("pickled/undersampling_table.pickle", "wb") as ut:
            pickle.dump(undersampling_table,ut)

    return rand2sample, undersampling_table
     
