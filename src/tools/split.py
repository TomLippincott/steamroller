#!/usr/bin/env python

import gzip
import codecs
import random
from .io import reader, writer

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input")
    parser.add_argument("--total_file", dest="total_file")
    parser.add_argument("--training_size", dest="training_size", type=float)
    parser.add_argument("--testing_size", dest="testing_size", type=float)
    parser.add_argument("--train", dest="train")
    parser.add_argument("--test", dest="test")
    options = parser.parse_args()

    with reader(gzip.open(options.total_file)) as ifd:
        total = int(ifd.read().strip())

    indices = list(range(total))
    random.shuffle(indices)

    if options.training_size <= 1:
        options.training_size = int(options.training_size * total)
        if options.testing_size <= 1:
            options.testing_size = int(options.testing_size * total)
    else:
        options.training_size = int(options.training_size)
        options.testing_size = int(options.testing_size)
        
    with writer(gzip.open(options.train, "w")) as ofd:
        ofd.write("\n".join([str(i) for i in indices[0:options.training_size]]))

    with writer(gzip.open(options.test, "w")) as ofd:
        ofd.write("\n".join([str(i) for i in indices[options.training_size:options.training_size+options.testing_size]]))
        

