import gzip
from sklearn import naive_bayes, metrics
from sklearn.feature_extraction import DictVectorizer
import pickle
import codecs
import logging
from itertools import chain
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
        data = read_data(options.input, options.train)
        instances, labels = [], []        
        for cid, label, text in read_data(options.input, options.train):
            instances.append(dict(sum([extract_character_ngrams(text, options.max_ngram) for n in range(1, options.max_ngram + 1)], [])))
            labels.append(label)
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(instances)
        label_lookup = {}
        classifier = naive_bayes.MultinomialNB()
        for l in labels:
            label_lookup[l] = label_lookup.get(l, len(label_lookup))
        logging.info("Training with %d instances, %d labels", len(instances), len(label_lookup))
        classifier.fit(X, [label_lookup[l] for l in labels])
        with open(options.output, "w") as ofd:
            pickle.dump((classifier, dv, label_lookup), ofd)            
    # testing
    elif options.test and options.model and options.output and options.input:
        with open(options.model) as ifd:
            classifier, dv, label_lookup = pickle.load(ifd)
        instances, gold = [], []
        for cid, label, text in read_data(options.input, options.test):
            instances.append(dict(sum([extract_character_ngrams(text, options.max_ngram) for n in range(1, options.max_ngram + 1)], [])))
            gold.append((cid, label))
        logging.info("Testing with %d instances, %d labels", len(instances), len(label_lookup))
        X = dv.transform(instances)
        inv_label_lookup = {v : k for k, v in label_lookup.iteritems()}
        data = {}
        order = [inv_label_lookup[i] for i in range(len(inv_label_lookup))]
        for probs, (cid, g) in zip([classifier.predict_log_proba(x) for x in X], gold):
            data[cid] = (g, {k : v for k, v in zip(order, probs.flatten())})
        write_probabilities(data, options.output)
    else:
        print "ERROR: you must specify --input and --output, and either --train or --test and --model!"
