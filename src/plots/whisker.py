import pandas
import numpy
import gzip
import csv
import matplotlib as mpl
mpl.use('Agg')
from plotnine import ggplot, aes, geom_boxplot, coord_flip, theme, geom_violin, geom_point, scale_x_continuous, ylab, xlab, stat_smooth, geom_line, ggtitle, guide_legend, theme_minimal, theme_void, theme_update, element_text, labs, geom_col, lims
from plotnine.themes.themeable import legend_title
from plotnine.guides import guide_legend, guides
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
    parser.add_argument("-i", "--input", dest="input")
    parser.add_argument("--x", dest="x")
    parser.add_argument("--y", dest="y")
    parser.add_argument("--color", dest="color")
    parser.add_argument("--xlabel", dest="xlabel")
    parser.add_argument("--ylabel", dest="ylabel")
    parser.add_argument("--color_label", dest="color_label")
    options = parser.parse_args()

    items = {}
    data = {}
    all_models = set()
    all_tasks = set()
    data = {}
    with gzip.open(options.input) as ifd:
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
    #print df
    x = (ggplot(df, aes("factor(%s)" % (options.x), options.y, color="factor(%s)" % (options.color)))) + \
        ggtitle(options.title.strip("'")) + \
        ylab(options.ylabel.strip("'")) + \
        xlab(options.xlabel.strip("'")) + \
        labs(color=options.color_label.strip("'")) + \
        geom_col(show_legend=False) + \
        lims(y=(0.0, 1.0))
    x.save(options.output)

    #theme(legend_title=element_text("")) + \
