#!/usr/bin/env python

import gzip
from sklearn import naive_bayes, linear_model, svm, metrics, ensemble, dummy
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
import pickle
import codecs
import logging
from itertools import chain
from steamroller.tools.io import read_data, read_subset
#write_probabilities, writer, reader, extract_character_ngrams
import numpy as np
from numpy import logspace

models = {
    "sgd" : (linear_model.SGDClassifier, {"penalty" : "l2", "alpha" : 1.0, "max_iter" : 1000, "loss" : "log"}, []), #[{"alpha" : [1.0]}]), #logspace(-4, 4, 7)}]),
    "naive_bayes" : (naive_bayes.MultinomialNB, {}, {"alpha" : [0.0, 1.0]}),
    "svm" : (svm.SVC, {"kernel" : "linear", "probability" : True}, {"C" : logspace(-4, 4, 9)}),
    #"svm" : (svm.SVC, {}, [{"kernel" : ["linear"], "C" : logspace(-3, 3, 7)},
    #                       {"kernel" : ["rbf"], "C" : logspace(-3, 3, 7), "gamma" : logspace(-9, 3, 13)}],
    #),                   
    "logistic_regression" : (linear_model.LogisticRegression, {"class_weight" : "balanced"}, {"penalty" : ["l1", "l2"], "C" : logspace(-4, 4, 9)}),
    "random_forest" : (ensemble.RandomForestClassifier, {"class_weight" : "balanced"}, {}),
    "prior" : (dummy.DummyClassifier, {"strategy" : "prior"}, {}),
}


if __name__ == "__main__":

    import argparse

    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input")
    parser.add_argument("--model_type", dest="model_type", choices=models.keys())
    #parser.add_argument("--label_field", dest="label_field", default="attribute")
    #parser.add_argument("--kbest", dest="kbest", type=int, default=-1)
    parser.add_argument("--train", dest="train")
    parser.add_argument("--test", dest="test")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--output", dest="output")
    parser.add_argument("--iterations", dest="iterations", type=int, default=50)    
    #parser.add_argument("--batch_size", dest="batch_size", type=int, default=None)
    options = parser.parse_args()
    
    # training
    if options.train and options.output and options.input:
        pass
        # if options.batch_size:
        #     label_lookup = {}
        #     feature_lookup = {}
        #     #for cid, label, text in read_data(options.input, options.train, tag_type=options.tag_type):
        #     #    for f in dict(sum([extract_character_ngrams(text, n) for n in range(1, options.max_ngram + 1)], [])).keys():
        #     #        feature_lookup[f] = feature_lookup.get(f, len(feature_lookup))
        #     #    label_lookup[label] = label_lookup.get(label, len(label_lookup))
        #     #logging.info("Performed initial pass...")
        #     items = []
        #     #for j in read_subset(args.input, args.train, label_field):
        #     #    items.append(j)
        #     dv = DictVectorizer(sparse=True)
        #     X = dv.fit_transform([feature_lookup])
        #     classifier_class, args, hypers = models[options.model_type]
        #     classifier = classifier_class(**args)
        #     for i in range(options.iterations):
        #         instances = []
        #         labels = []
        #         for cid, label, text in read_data(options.input, options.train, tag_type=options.tag_type):
        #             instances.append(dict(sum([extract_character_ngrams(text, n) for n in range(1, options.max_ngram + 1)], [])))
        #             labels.append(label)
        #             if len(instances) >= options.batch_size:
        #                 logging.info("Running partial fit...")
        #                 X = dv.transform(instances)
        #                 classifier.partial_fit(X, [label_lookup[l] for l in labels], classes=[i for i in label_lookup.values()])
        #                 instances = []
        #                 labels = []
        #         if len(instances) < options.batch_size:
        #             logging.info("Running partial fit...")
        #             X = dv.transform(instances)
        #             classifier.partial_fit(X, [label_lookup[l] for l in labels], classes=[i for i in label_lookup.values()])
        #             instances = []
        #             labels = []
        #     with gzip.open(options.output, "w") as ofd:
        #         pickle.dump((classifier, [dv], label_lookup), ofd)
        # else:
        #     instances, labels = [], []
        #     for cid, label, text in read_data(options.input, options.train, tag_type=options.tag_type):
        #         instances.append(dict(sum([extract_character_ngrams(text, n) for n in range(1, options.max_ngram + 1)], [])))
        #         labels.append(label)
        #     logging.info("Loaded %d instances", len(instances))
        #     dv = DictVectorizer(sparse=True)
        #     X = dv.fit_transform(instances)
        #     fs = SelectKBest(k=options.kbest if options.kbest > 0 else X.shape[1])
        #     X = fs.fit_transform(X, labels)
        #     label_lookup = {}
        #     classifier_class, args, hypers = models[options.model_type]
        #     classifier = classifier_class(**args) # GridSearchCV(classifier_class(**args), hypers)
        #     for l in labels:
        #         label_lookup[l] = label_lookup.get(l, len(label_lookup))
        #     logging.info("Training with %d instances, %d features, %d labels", len(instances), len(fs.get_support(indices=True)), len(label_lookup))
        #     classifier.fit(X, [label_lookup[l] for l in labels])
        #     with gzip.open(options.output, "w") as ofd:
        #         pickle.dump((classifier, [dv, fs], label_lookup), ofd)            
    # testing
    elif options.test and options.model and options.output and options.input:
        pass
        # with gzip.open(options.model) as ifd:
        #     classifier, ts, label_lookup = pickle.load(ifd)
        # instances, gold = [], []
        # inv_label_lookup = {v : k for k, v in label_lookup.items()}
        # order = [inv_label_lookup[i] for i in range(len(inv_label_lookup))]
        # data = {}
        # for cid, label, text in read_data(options.input, options.test, tag_type=options.tag_type):
        #     instances.append(dict(sum([extract_character_ngrams(text, n) for n in range(1, options.max_ngram + 1)], [])))
        #     gold.append((cid, label))

        #     logging.info("Testing with %d instances, %d labels", len(instances), len(label_lookup))
        #     X = instances
        #     for t in ts:
        #         X = t.transform(X)
        #     if hasattr(classifier, "predict_log_proba"):
        #         for probs, (cid, g) in zip(classifier.predict_log_proba(X), gold):
        #             data[cid] = (g, {k : v for k, v in zip(order, probs.flatten())})
        #     else:
        #         for pred, (cid, g) in zip(classifier.predict(X), gold):
        #             probs = [0.0 if i == pred else float("-inf") for i in range(len(order))]
        #             data[cid] = (g, {k : v for k, v in zip(order, probs)})
        #     instances = []
        #     gold = []

        # #X = instances
        # #for t in ts:
        # #    X = t.transform(X)
        

        
        # write_probabilities(data, options.output)
    else:
        print("ERROR: you must specify --input and --output, and either --train or --test and --model!")
