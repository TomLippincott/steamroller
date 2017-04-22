import readline
import rpy2
import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2.robjects.lib.ggplot2 import ggplot
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import numpy
import gzip
import csv

grdevices = importr('grDevices')

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument("-f", "--field", dest="field")
    parser.add_argument(nargs="+", dest="inputs")
    options = parser.parse_args()

    items = {}
    data = {}
    all_models = set()
    all_tasks = set()
    data = []
    for s in options.inputs:
        with gzip.open(s) as ifd:
            #c = 
            #fields = ifd.readline().strip().split("\t")
            for row in csv.DictReader(ifd, delimiter="\t"): # in ifd:
                #task, model, size, fold, score = line.strip().split("\t")
                #print row
                data.append((row["task"], row["model"], int(row["size"]), int(row["fold"]), float(row[options.field])))
                
    df = ro.DataFrame({
        "task" : ro.StrVector([x[0] for x in data]),
        "model" : ro.StrVector([x[1] for x in data]),
        "size" : ro.IntVector([x[2] for x in data]),
        "fold" : ro.IntVector([x[3] for x in data]),
        "score" : ro.FloatVector([x[4] for x in data]),
    })
    grdevices.png(file=options.output, width=768, height=512)
    
    x = ggplot(df) + \
        ggplot2.aes_string(x="factor(size)", y="score", col="factor(model)") + \
        ggplot2.geom_boxplot() + \
        ggplot2.theme_bw() + \
        ggplot2.theme(text=ggplot2.element_text(size=20, family="serif"),
                      #legend_position="none",
                      legend_title=ggplot2.element_blank(), plot_title=ggplot2.element_text(hjust=.5, size=20, family="serif")) + \
        ggplot2.ggtitle(row["task"].title()) + \
        ggplot2.labs(y=options.field, x="Document Count (logarithmic)")
        #

    x.plot()
    grdevices.dev_off()
