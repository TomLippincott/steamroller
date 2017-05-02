#!/usr/bin/env python

import gzip
import pickle
import codecs
import logging
from itertools import chain
from data_io import read_data, write_probabilities, writer, reader, extract_character_ngrams
import numpy


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

    numpy.random.seed()
    
    # training
    if options.train and options.output and options.input:
        data = read_data(options.input, options.train)
        instances, labels = [], []
        counts = {}
        for cid, label, text in read_data(options.input, options.train):
            counts[label] = counts.get(label, 0) + 1
        total = float(sum(counts.values()))
        prior = {k : v / total for k, v in counts.iteritems()}
        with gzip.open(options.output, "w") as ofd:
            pickle.dump(prior, ofd)
            
    # testing
    elif options.test and options.model and options.output and options.input:
        with gzip.open(options.model) as ifd:
            prior = list(pickle.load(ifd).iteritems())
        instances, gold = [], []
        for cid, label, text in read_data(options.input, options.test):
            instances.append(dict(sum([extract_character_ngrams(text, options.max_ngram) for n in range(1, options.max_ngram + 1)], [])))
            gold.append((cid, label))
        data = {}
        for cid, g in gold:
            probs = {k : float("-inf") for k, _ in prior}
            i = numpy.random.multinomial(1, [p for _, p in prior]).tolist().index(1)
            probs[prior[i][0]] = 0.0
            data[cid] = (g, probs)
        write_probabilities(data, options.output)
