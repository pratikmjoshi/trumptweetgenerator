import numpy as np
import sys
import os
import keras
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout,Bidirectional,TimeDistributed,Activation


def get_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size,100))
    model.add(Bidirectional(LSTM(50,return_sequences = True)))
    model.add(LSTM(50,return_sequences = True))
    model.add(LSTM(50,return_sequences = True))
    model.add(LSTM(50,return_sequences = True))

    model.add(TimeDistributed(vocab_size))
    model.add(Activation('softmax'))

    model.compile(loss = 'categorical_crossentropy',optimizer='rmsprop')

    return model
