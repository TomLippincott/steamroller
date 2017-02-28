import gzip
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
import pickle
import codecs
import logging
from itertools import chain

writer = codecs.getwriter("utf-8")
reader = codecs.getreader("utf-8")

def _extract_character_ngrams(text, n):
    stack = ["NULL" for _ in range(n)]
    ngrams = {}
    for c in text:
        stack = stack[1:]
        stack.append(c)
        ngrams[tuple(stack)] = ngrams.get(tuple(stack), 0) + 1
    return list(ngrams.iteritems())

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
        with reader(gzip.open(options.train)) as ifd:
            indices = [int(l.strip()) for l in ifd]
        indices = set(indices)
        instances, labels = [], []        
        with reader(gzip.open(options.input)) as ifd:
            for i, line in enumerate(ifd):
                if i in indices:
                    cid, label, text = line.strip().split("\t")
                    instances.append(dict(sum([_extract_character_ngrams(text, options.max_ngram) for n in range(1, options.max_ngram + 1)], [])))
                    labels.append(label)
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(instances)
        label_lookup = {}
        classifier = linear_model.LogisticRegression(class_weight="balanced", solver="sag")
        for l in labels:
            label_lookup[l] = label_lookup.get(l, len(label_lookup))
        logging.info("Training with %d instances, %d labels", len(instances), len(label_lookup))
        classifier.fit(X, [label_lookup[l] for l in labels])
        with gzip.open(options.output, "w") as ofd:
            pickle.dump((classifier, dv, label_lookup), ofd)            

    # testing
    elif options.test and options.model and options.output and options.input:
        with gzip.open(options.model) as ifd:
            classifier, dv, label_lookup = pickle.load(ifd)
        with reader(gzip.open(options.test)) as ifd:
            indices = [int(l.strip()) for l in ifd]
        indices = set(indices)
        instances, gold = [], []
        with reader(gzip.open(options.input)) as ifd:
            for i, line in enumerate(ifd):
                if i in indices:
                    cid, label, text = line.strip().split("\t")
                    instances.append(dict(sum([_extract_character_ngrams(text, options.max_ngram) for n in range(1, options.max_ngram + 1)], [])))
                    gold.append((cid, label))
        logging.info("Testing with %d instances, %d labels", len(instances), len(label_lookup))
        X = dv.transform(instances)
        inv_label_lookup = {v : k for k, v in label_lookup.iteritems()}
        y = [classifier.predict_log_proba(x) for x in X]
        with writer(gzip.open(options.output, "w")) as ofd:
            ofd.write("\t".join(["ID", "GOLD"] + [inv_label_lookup[i] for i in range(len(inv_label_lookup))]) + "\n")
            for probs, (cid, gold) in zip(y, gold):
                ofd.write("\t".join([cid, gold] + ["%f" % x for x in probs.flatten()]) + "\n")
    
    else:
        print "ERROR: you must specify --input and --output, and either --train or --test and --model!"
