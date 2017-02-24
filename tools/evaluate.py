import gzip
import numpy
import re

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument(nargs="+", dest="inputs")
    options = parser.parse_args()

    accuracies = {}
    for f in options.inputs:
        task, model, size, fold = re.match(r"^work/(.*)_(.*)_(.*)_(.*)_results.txt.gz$", f).groups()
        size = int(size)
        fold = int(fold)
        vals = {}
        correct, incorrect = 0.0, 0.0
        truepos = 0.0
        falsepos = 0.0
        trueneg = 0.0
        falseneg = 0.0
        with gzip.open(f) as ifd:
            fields = ifd.readline().strip().split("\t")            
            for l in ifd:
                toks = l.strip().split("\t")
                cid, gold = toks[0:2]
                scores = {k : float(v) for k, v in zip(fields[2:], toks[2:])}
                guess = sorted([(v, k) for k, v in scores.iteritems()])[-1][1]
                if gold == guess:
                    correct += 1.0
                else:
                    incorrect += 1.0
                        
            key = (task, size, model, fold)
            accuracies[key] = accuracies.get(key, []) + [correct / (correct + incorrect)]

    with gzip.open(options.output, "w") as ofd:
        for (task, size, model, fold), a in sorted(accuracies.iteritems()):
            ss = numpy.array(a)
            ofd.write("\t".join([task, model, str(size), str(fold), "%.3f" % (ss.mean())]) + "\n")
