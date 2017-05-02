#!/usr/bin/env python

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
        task, model, size, fold = re.match(r"^work/(.*)_(.*)_(.*)_(.*)\.model.*$", f).groups()
        size = int(size)
        fold = int(fold)
        key = (task, size, model, fold)
        sizes[key] = os.lstat(f).st_size
        
    with gzip.open(options.output, "w") as ofd:
        c = csv.DictWriter(ofd, fieldnames=["task", "size", "model", "fold", "Gigabytes"], delimiter="\t")
        c.writeheader()
        for (task, size, model, fold), s in sorted(sizes.iteritems()):
            c.writerow({"task" : task, "model" : model, "size" : size, "fold" : fold, "Gigabytes" : s / 1000000000.0})
