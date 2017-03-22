import gzip
import numpy
import re
from sklearn.metrics import f1_score
import math

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input")
    options = parser.parse_args()
    
    scores = {}
    accuracies = {}

    vals = {}
    correct, incorrect = 0.0, 0.0
    truepos = 0.0
    falsepos = 0.0
    trueneg = 0.0
    falseneg = 0.0
    with gzip.open(options.input) as ifd:
        fields = ifd.readline().strip().split("\t")
        y_true = []
        y_pred = []
        for l in ifd:
            toks = l.strip().split("\t")
            cid, gold = toks[0:2]
            s = {k : float(v) for k, v in zip(fields[2:], toks[2:])}
            guess = sorted([(v, k) for k, v in s.iteritems()])[-1][1]
            #if "unk" not in [gold, guess]:
            y_true.append(gold)
            y_pred.append(guess)
            if gold == guess:
                correct += 1.0
            else:
                incorrect += 1.0
        print f1_score(y_true=y_true, y_pred=y_pred, average="macro")

    
