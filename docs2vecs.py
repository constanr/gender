#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from gensim.models import Doc2Vec

from pprocess_twitter import tokenize
from nltk import word_tokenize
import string

###
### Change for specific dir
###

models_path = 'models/pan16-author-profiling-training-dataset-{}-2016-04-25/'

models = {
    'english': 'D2Vmodel',
    'spanish': 'D2Vmodel',
    'dutch': 'D2Vmodel2'
}

###

noise = set(string.punctuation)-set('¡!¿?,.:') # > and < are removed also
noise = {ord(c):None for c in noise}
def normalize_text(text):
    t = tokenize(text)
    t = t.lower().translate(noise)
    return word_tokenize(t)

def check_lang(lang):
    langs = set(models.keys())
    if lang in langs:
        return True
    return False

def load_model(language, models_path, models):
    if check_lang:
        path = models_path.format(language) + models[language]
        print path
        model = Doc2Vec.load(path)
        assert model.docvecs.count > 0
        return model
    else:
        return None

def extract_vector(doc, model):
        doc = normalize_text(doc)
        v = model.infer_vector(doc)
        v = np.nan_to_num(v)
        return v
