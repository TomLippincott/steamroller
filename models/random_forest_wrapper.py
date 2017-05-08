#!/usr/bin/env python

import gzip
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import pickle
import codecs
import logging
from itertools import chain
import numpy
from data_io import read_data, write_probabilities, writer, reader, extract_character_ngrams

if __name__ == "__main__":

    import argparse

    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input")
    parser.add_argument("--train", dest="train")
    parser.add_argument("--test", dest="test")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--output", dest="output")
    parser.add_argument("--max_ngram", dest="max_ngram", type=int, default=4)    
    options = parser.parse_args()
    
    # training
    if options.train and options.output and options.input:
        instances, labels = [], []        
        for cid, label, text in read_data(options.input, options.train):
            instances.append(dict(sum([extract_character_ngrams(text, options.max_ngram) for n in range(1, options.max_ngram + 1)], [])))
            labels.append(label)
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(instances)
        scaler = preprocessing.StandardScaler(with_mean=False).fit(X)
        label_lookup = {}
        #C_range = numpy.logspace(-3, 3, 6)
        #gamma_range = numpy.logspace(-3, 3, 6)
        #param_grid = dict(gamma=gamma_range, C=C_range)
        #param_grid = dict(C=C_range)        
        #cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2)
        for l in labels:
            label_lookup[l] = label_lookup.get(l, len(label_lookup))
        y = [label_lookup[l] for l in labels]
        #grid = GridSearchCV(svm.SVC(kernel="linear"), param_grid=param_grid, cv=cv)
        #try:
        #    grid.fit(X, y)
        #    C = grid.best_params_["C"]
        #except:
        #    C = 1.0
        #print grid.best_params_
        classifier = RandomForestClassifier(class_weight="balanced")
        #svm.SVC(kernel="linear", probability=True, C=C) #, gamma=grid.best_params_["gamma"])

        logging.info("Training with %d instances, %d labels", len(instances), len(label_lookup))
        classifier.fit(scaler.transform(X), [label_lookup[l] for l in labels])
        with gzip.open(options.output, "w") as ofd:
            pickle.dump((classifier, scaler, dv, label_lookup), ofd)            

    # testing
    elif options.test and options.model and options.output and options.input:
        with gzip.open(options.model) as ifd:
            classifier, scaler, dv, label_lookup = pickle.load(ifd)
        instances, gold = [], []
        for cid, label, text in read_data(options.input, options.test):
            instances.append(dict(sum([extract_character_ngrams(text, options.max_ngram) for n in range(1, options.max_ngram + 1)], [])))
            gold.append((cid, label))
        logging.info("Testing with %d instances, %d labels", len(instances), len(label_lookup))
        X = scaler.transform(dv.transform(instances))
        inv_label_lookup = {v : k for k, v in label_lookup.iteritems()}
        data = {}
        order = [inv_label_lookup[i] for i in range(len(inv_label_lookup))]
        for probs, (cid, g) in zip([classifier.predict_log_proba(x) for x in X], gold):
            data[cid] = (g, {k : v for k, v in zip(order, probs.flatten())})
        write_probabilities(data, options.output)
