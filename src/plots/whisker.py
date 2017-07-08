import pandas
import numpy
import gzip
import csv
import matplotlib as mpl
mpl.use('Agg')
from plotnine import ggplot, aes, geom_boxplot, coord_flip, theme, geom_violin, geom_point, scale_x_continuous, ylab, xlab, stat_smooth, geom_line, ggtitle
import re

def maybe_float(s):
    try:
        return float(s)
    except:
        return None

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--title", dest="title", default="Box Plot")
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument("-f", "--field", dest="field")
    parser.add_argument(nargs="+", dest="inputs")
    options = parser.parse_args()

    items = {}
    data = {}
    all_models = set()
    all_tasks = set()
    data = {}
    for s in options.inputs:
        with gzip.open(s) as ifd:
            for row in csv.DictReader(ifd, delimiter="\t"):
                for k, v in row.iteritems():
                    data[k] = data.get(k, [])
                    data[k].append(v)
    
    for k in data.keys():
        floats = [maybe_float(x) for x in data[k]]
        if all([re.match(r"^\d+$", x) for x in data[k]]):
            data[k] = [int(x) for x in data[k]]
        elif all(floats):
            data[k] = floats

    df = pandas.DataFrame(data)

    x = (ggplot(df, aes("factor(size)", options.field, color="factor(model)"))) + \
        ggtitle(options.title) + \
        scale_x_continuous(trans="log2") + \
        ylab(options.field) + \
        xlab("Training data points") + \
        geom_boxplot()

    x.save(options.output)
