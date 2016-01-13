#coding=utf-8
'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys
import os.path

import pymongo
from pymongo.read_preferences import ReadPreference
from keras.models import model_from_json
# import redis


conn = pymongo.MongoReplicaSetClient("h44:27017, h213:27017, h241:27017", replicaSet="myset",
                                     read_preference=ReadPreference.SECONDARY)
model_fname = '2lstm512.json'
model_weight_fname = '2lstm512_weights.h5'
input_data = 'data.txt'
reload(sys)
sys.setdefaultencoding('utf-8')

def prepare_data():
    isExist = os.path.isfile(input_data)
    if isExist:
        return unicode(open(input_data).read()).lower()  #to unicode with chinese character
    print('Fetching data...')
    result_list = conn['news_ver2']['googleNewsItem'].find({"isOnline": 1}).\
        sort([("createTime", pymongo.DESCENDING)]).limit(2).batch_size(200)
    text = ''
    for result in result_list:
        text += result['title']
        text += '\n'
        text += result['text']
    open(input_data, 'w').write(text.encode('utf-8'))
    return text.lower()


def old_one():
    path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
    text = open(path).read().lower()
    return text

text = prepare_data().encode('utf-8')
print('corpus length:', len(text))
chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


def prepare_model():
    isExist = os.path.isfile(model_fname)
    if not isExist:
        # build the model: 2 stacked LSTM
        print('Build model...')
        model = Sequential()
        model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
        model.add(Dropout(0.2))
        model.add(LSTM(512, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    else:
        model = model_from_json(open(model_fname).read())
        if os.path.isfile(model_weight_fname):
            model.load_weights(model_weight_fname)
    return model


def save_model(model):
    print('Saving...')
    json_string = model.to_json()
    open(model_fname, 'w').write(json_string)
    model.save_weights(model_weight_fname)

g_model = prepare_model()


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 2000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    g_model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = g_model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

    #save_model(g_model)
