#!/usr/bin/env python

from valid.model import Compressor, Classifier
import gzip
import codecs
from itertools import chain
import pickle
import logging
import math
import numpy
from steamroller.tools.io import read_data, write_probabilities, writer, reader

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input")
    parser.add_argument("--train", dest="train")
    parser.add_argument("--tag-type", dest="tag_type", default="attribute")
    parser.add_argument("--dev", dest="dev")
    parser.add_argument("--test", dest="test")
    parser.add_argument("--prior", dest="prior", default=False, action="store_true")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--output", dest="output")
    parser.add_argument("--max_ngram", dest="max_ngram", type=int, default=4)
    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    if options.train and options.output and options.input:
        counts = {}
        model = Classifier(options.max_ngram)
        for cid, label, text in read_data(options.input, options.train, tag_type=options.tag_type):
            counts[label] = counts.get(label, 0) + 1
            model.train(label, text)
        
        with gzip.open(options.output, "w") as ofd:
            pickle.dump((model, counts), ofd)

    # testing
    elif options.test and options.model and options.output and options.input:
        instances, gold = [], []
        for cid, label, text in read_data(options.input, options.test, tag_type=options.tag_type):
            instances.append(text)
            gold.append((cid, label))
        
        with gzip.open(options.model) as ifd:
            model, counts = pickle.load(ifd)
        total = sum(counts.values())
        prior = {k : math.log(v / float(total)) for k, v in counts.items()}
        codes = model.compressors.keys()
        logging.info("Testing with %d instances, %d labels", len(instances), len(codes))
        data = {}
        for (cid, label), text in zip(gold, instances):
            probs = {k : v + (prior[k] if options.prior else 0.0) for k, v in model.classify(text).items()}
            #total = reduce(numpy.logaddexp, probs.values())
            data[cid] = (label, probs)
        write_probabilities(data, options.output)
