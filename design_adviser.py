from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from decimal import Decimal, ROUND_HALF_UP
import MeCab
import gensim
import json


global attr, color_estimator, graph, color_estimators, word_vectors
curdir = os.path.dirname(__file__)
with open(os.path.join(curdir, 'config.json')) as f:
    config_dict = json.load(f)
word_vectors = pickle.load(open(
    os.path.join(curdir,
    'pkl_objects',
    'word_vectors.pkl'), 'rb'
))
graph = tf.get_default_graph()
attr = load_model(os.path.join(curdir, 'h5_objects/attr.h5'))
color_estimator = load_model(os.path.join(curdir, 'h5_objects/color_sort.h5'))
color_list = ['color1', 'color2', 'color3', 'color4', 'color5']
color_estimators = [load_model(os.path.join(curdir, 'h5_objects/{color}.h5'.format(color=color))) for color in color_list]

def vectorize(text, input_values):
    text_vec = get_text_vec(text)
    input_vec = np.array([1.0 if val in input_values else 0.0 for val in config_dict['input']])
    print(text_vec.shape, input_vec.shape)
    return np.append(text_vec, input_vec)

def get_text_vec(text):
    #print(text)
    if not text:
        return np.zeros(300)
    mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    mecab.parse('')#文字列がGCされるのを防ぐ
    node = mecab.parseToNode(text)
    vecs = []
    while node:
        #単語を取得
        if node.surface in word_vectors.index2word:
            vecs.append(word_vectors[node.surface])
        #次の単語に進める
        node = node.next
    vec = np.mean(np.array(vecs), axis=0)
    print(vec)
    return vec

def predict_attr(x):
    with graph.as_default():
        return attr.predict(x)[0]

def predict_color(x):
    with graph.as_default():
        return color_estimator.predict(x)[0]