#!/usr/bin/env python

import gzip
import codecs
import random
import csv
from io import reader, writer

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument(nargs="+", dest="inputs")
    options = parser.parse_args()

    key_fields = ["task", "fold", "model", "size"]
    fields = set()
    rows = {}
    for ifile in options.inputs:
        with gzip.open(ifile) as ifd:
            for entry in csv.DictReader(ifd, delimiter="\t"):
                key = tuple([entry[f] for f in key_fields])
                rows[key] = rows.get(key, {})
                for k, v in entry.iteritems():
                    fields.add(k)
                    if k not in key_fields:
                        rows[key][k] = v
                    

    with gzip.open(options.output, "w") as ofd:
        c = csv.DictWriter(ofd, fieldnames=fields, delimiter="\t")
        c.writeheader()
        for fs, r in rows.iteritems():
            c.writerow({k : v for k, v in zip(key_fields, fs) + list(r.iteritems())})

