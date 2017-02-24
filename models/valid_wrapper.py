from valid.model import Compressor, LidClassifier
import gzip
import codecs
from itertools import chain
import pickle
import logging

writer = codecs.getwriter("utf-8")
reader = codecs.getreader("utf-8")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input")
    parser.add_argument("--train", dest="train")
    parser.add_argument("--dev", dest="dev")
    parser.add_argument("--test", dest="test")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--output", dest="output")
    parser.add_argument("--max_ngram", dest="max_ngram", type=int, default=4)
    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    if options.train and options.output and options.input:
        with reader(gzip.open(options.train)) as ifd:
            indices = [int(l.strip()) for l in ifd]
        indices = set(indices)
        instances, labels = [], []
        compressors = {}
        with reader(gzip.open(options.input)) as ifd:
            for i, line in enumerate(ifd):
                if i in indices:
                    cid, label, text = line.strip().split("\t")
                    compressors[label] = compressors.get(label, Compressor(label, options.max_ngram))
                    compressors[label].add(text)

        model = LidClassifier()
        for l, c in compressors.iteritems():
            model.add(c, l)

        with gzip.open(options.output, "w") as ofd:
            pickle.dump(model, ofd)

    # testing
    elif options.test and options.model and options.output and options.input:
        with reader(gzip.open(options.test)) as ifd:
            indices = [int(l.strip()) for l in ifd]
        indices = set(indices)
        instances, gold = [], []
        with reader(gzip.open(options.input)) as ifd:
            for i, line in enumerate(ifd):
                if i in indices:
                    cid, label, text = line.strip().split("\t")
                    instances.append(text)
                    gold.append((cid, label))
        
        with gzip.open(options.model) as ifd:
            model = pickle.load(ifd)
        codes = model.compressors.keys()
        logging.info("Testing with %d instances, %d labels", len(instances), len(codes))            
        with writer(gzip.open(options.output, "w")) as ofd:
            ofd.write("\t".join(["DOC", "USER", "GOLD"] + codes) + "\n")
            for (cid, label), text in zip(gold, instances):
                probs = model.classify(text)
                ofd.write("\t".join([cid, label] + ["%f" % probs[l] for l in codes]) + "\n")

