import csv
import os.path
import os
import gzip
import re


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument(dest="inputs", nargs="+")
    options = parser.parse_args()

    sizes = {}
    for f in options.inputs:
        task, model, size, fold = re.match(r"^work/(.*)_(.*)_(.*)_(.*)\.model$", f).groups()
        size = int(size)
        fold = int(fold)
        
        # vals = {}
        # correct, incorrect = 0.0, 0.0
        # truepos = 0.0
        # falsepos = 0.0
        # trueneg = 0.0
        # falseneg = 0.0
        # with gzip.open(f) as ifd:
        #     fields = ifd.readline().strip().split("\t")
        #     y_true = []
        #     y_pred = []
        #     for l in ifd:
        #         toks = l.strip().split("\t")
        #         cid, gold = toks[0:2]
        #         s = {k : float(v) for k, v in zip(fields[2:], toks[2:])}
        #         guess = sorted([(v, k) for k, v in s.iteritems()])[-1][1]
        #         y_true.append(gold)
        #         y_pred.append(guess)
        #         if gold == guess:
        #             correct += 1.0
        #         else:
        #             incorrect += 1.0
                        
        key = (task, size, model, fold)
        sizes[key] = os.lstat(f).st_size
        #     scores[key] = scores.get(key, []) + [f1_score(y_true=y_true, y_pred=y_pred, average="macro")]
        #     accuracies[key] = accuracies.get(key, []) + [correct / (correct + incorrect)]

    with gzip.open(options.output, "w") as ofd:
        c = csv.DictWriter(ofd, fieldnames=["task", "size", "model", "fold", "Size"], delimiter="\t")
        c.writeheader()
        for (task, size, model, fold), s in sorted(sizes.iteritems()):
            c.writerow({"task" : task, "model" : model, "size" : size, "fold" : fold, "Size" : s})
