import numpy as np

def get_vector(word_list,model):
    vocab = model.wv.vocab
    #build array
    res = np.zeros([128])
    count=0
    for word in word_list:
        if word in vocab:   
            res += model.wv[word]
            count += 1
    return res/count