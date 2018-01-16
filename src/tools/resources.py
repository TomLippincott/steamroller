#!/usr/bin/env python

import gzip
import math
import csv
import pickle
import re

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument("-s", "--stage", dest="stage")
    parser.add_argument(nargs="+", dest="inputs")
    options = parser.parse_args()
    
    vals = {}
    for f in options.inputs:
        task, model, size, fold, _ = re.match(r"^work/(.*?)_(.*?)_(.*?)_(.*?)(\.|_).*txt$", f).groups()
        #size = int(size)
        #fold = int(fold)
        with gzip.open(f) as ifd:
            u = pickle.load(ifd)
            key = (task, size, model, fold)
            #res = {k : float(v) for k, v in u} #pattern.search(ifd.read()).groupdict().items()}
            
            vals[key] = (u.ru_maxrss, u.ru_utime)

    with open(options.output, "w") as ofd:
        c = csv.DictWriter(ofd, fieldnames=["task", "size", "model", "fold", "%s_Memory" % (options.stage), "%s_CPU" % (options.stage)], delimiter="\t")
        c.writeheader()
        for (task, size, model, fold), (mem, cpu) in sorted(vals.items()):
            c.writerow({"task" : task, "model" : model, "size" : size, "fold" : fold, "%s_Memory" % (options.stage) : mem / 1000000.0, "%s_CPU" % (options.stage) : cpu})

