#!/usr/bin/env python

import gzip
import codecs
import random
from concrete.util.file_io import CommunicationReader
from .io import reader, writer

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input")
    parser.add_argument("-o", "--output", dest="output")    
    options = parser.parse_args()

    with writer(gzip.open(options.output, "w")) as ofd:
        for i, c in enumerate(CommunicationReader(options.input)):
            ofd.write("%d\n" % (i))

