from model import get_model
from keras.models import load_model
import numpy as np
import os
import sys
import itertools

SEQ_LEN = 10

def main():
    train = input("Skip training?:")

    tweets = np.load('tweets.npy')
    print("Number of tweets:{}".format(len(tweets)))
    print("Example tweet:{}".format(tweets[4]))

    vocab = sorted(list(set(itertools.chain.from_iterable(tweets))))
    index_to_word = dict(list(enumerate(vocab)))
    word_to_index = dict([(word,i) for i,word in enumerate(vocab)])


    print("Vocabulary size:{}".format(len(vocab)))
    if train == 'n':
        X_train,y_train,X_val,y_val,X_test,y_test = preprocess(tweets,vocab,index_to_word,word_to_index)
        print("Finished preprocess,starting training....")

        model = get_model(len(vocab))
        model.fit(X_train,y_train,batch_size=64,validation_data=(X_val,y_val))

        model.save('new_model.hdf5')
        print("Model performance:")
        print(model.evaluate(X_test,y_test))
    else:
        model = load_model('new_model.hdf5')
        cont = input("Continue?:")
        while(cont!='n'):
            print(generate_text(model,20,vocab,index_to_word))
            cont =  input("Continue?:")
    return





def preprocess(tweets,vocab,index_to_word,word_to_index):
    x = []
    y = []
    for tweet in tweets:
        for i in range(len(tweet)-SEQ_LEN):
            x_point = []
            for k in range(i,i+SEQ_LEN):
                temp = np.zeros(len(vocab))
                temp[word_to_index[tweet[k]]] = 1
                x_point.append(temp)
            x.append(x_point)
            y_temp = np.zeros(len(vocab))
            y_temp[word_to_index[tweet[i+SEQ_LEN]]] = 1
            y.append(y_temp)
    x = np.array(x)
    y = np.array(y)

    p = np.random.permutation(len(y))
    x = x[p]
    y = y[p]

    X_train = x[:int(3/4*len(y))]
    y_train = y[:int(3/4*len(y))]

    X_val = x[int(3/4*len(y))+1:int(7/8*len(y))]
    y_val = y[int(3/4*len(y))+1:int(7/8*len(y))]

    X_test = x[int(7/8*len(y))+1:]
    y_test = y[int(7/8*len(y))+1:]

    np.save('X_train.npy',X_train)
    np.save('y_train.npy',y_train)
    np.save('X_val.npy',X_val)
    np.save('y_val.npy',y_val)
    np.save('X_test.npy',X_test)
    np.save('y_test.npy',y_test)

    return X_train,y_train,X_val,y_val,X_test,y_test

def generate_text(model,length,vocab,index_to_word):
    randstartindex = np.random.randint(len(vocab))
    gen_string = [index_to_word[randstartindex]]
    input_data = np.zeros((1,length,len(vocab)))
    for i in range(length):
        input_data[0,i,:][randstartindex] = 1
        index = np.argmax(model.predict(input_data[:,:i+1,:])[0],1)
        gen_string.append(index_to_word[index[-1]])
    return ('').join(gen_string)


if __name__ == "__main__":
    main()
