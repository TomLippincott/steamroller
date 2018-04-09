import gzip
import numpy
import re
import math
import csv

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument(nargs="+", dest="inputs")
    options = parser.parse_args()
    
    scores = {}
    accuracies = {}
    for f in options.inputs:
        task, model, size, fold = re.match(r"^work/(.*?)_(.*?)_(.*?)_(.*?)_.*probabilities.txt.gz$", f).groups()
        #size = int(size)
        #fold = int(fold)
        vals = {}
        correct, incorrect = 0.0, 0.0
        truepos = 0.0
        falsepos = 0.0
        trueneg = 0.0
        falseneg = 0.0
        with gzip.open(f) as ifd:
            fields = ifd.readline().decode().strip().split("\t")
            y_true = []
            y_pred = []
            for l in ifd:
                toks = l.decode().strip().split("\t")
                cid, gold = toks[0:2]
                s = {k : float(v) for k, v in zip(fields[2:], toks[2:])}
                guess = sorted([(v, k) for k, v in s.items()])[-1][1]
                y_true.append(gold)
                y_pred.append(guess)
                if gold == guess:
                    correct += 1.0
                else:
                    incorrect += 1.0
                        
            key = (task, size, model, fold)
            accuracies[key] = accuracies.get(key, []) + [correct / (correct + incorrect)]
            
    with open(options.output, "w") as ofd:
        c = csv.DictWriter(ofd, fieldnames=["task", "size", "model", "fold", "Accuracy"], delimiter="\t")
        c.writeheader()
        for (task, size, model, fold), a in sorted(accuracies.items()):
            ss = numpy.array(a)
            c.writerow({"task" : task, "model" : model, "size" : size, "fold" : fold, "Accuracy" : ss.mean()})

    # with gzip.open(options.output, "w") as ofd:
    #     fieldnames=["task", "size", "model", "fold", "Accuracy"]
    #     #, delimiter="\t")
    #     #c = csv.DictWriter(ofd, 
    #     #c.writeheader()
    #     ofd.write(b"\t".join([f.encode() for f in fieldnames]) + b"\n")
    #     for (task, size, model, fold), a in sorted(accuracies.items()):
    #         ss = numpy.array(a)
    #         #c.writerow({"task" : task, "model" : model, "size" : size, "fold" : fold, "Accuracy" : ss.mean()})
    #         ofd.write(b"\t".join([str(v).encode() for v in [task, size, model, fold, ss.mean()]]) + b"\n")
