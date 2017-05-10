#!/usr/bin/env python

import gzip
import codecs
import random
from io import reader, writer

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input")
    parser.add_argument("--total_file", dest="total_file")
    parser.add_argument("--train_count", dest="train_count", type=int)
    parser.add_argument("--test_count", dest="test_count", type=int)
    parser.add_argument("--train", dest="train")
    parser.add_argument("--test", dest="test")
    options = parser.parse_args()

    with reader(gzip.open(options.total_file)) as ifd:
        total = int(ifd.read().strip())

    indices = range(total)
    random.shuffle(indices)
    
    with writer(gzip.open(options.train, "w")) as ofd:
        ofd.write("\n".join([str(i) for i in indices[0:options.train_count]]))

    with writer(gzip.open(options.test, "w")) as ofd:
        ofd.write("\n".join([str(i) for i in indices[options.train_count:options.train_count+options.test_count]]))


