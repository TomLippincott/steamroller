import readline
import rpy2
import rpy2.robjects.lib.ggplot2 as ggplot2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import numpy
import gzip

grdevices = importr('grDevices')

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument(nargs="+", dest="inputs")
    options = parser.parse_args()

    items = {}
    data = {}
    all_models = set()
    all_tasks = set()
    for s in options.inputs:
        with gzip.open(s) as ifd:
            for line in ifd:
                task, model, size, fold, acc = line.strip().split("\t")
                size = int(size)                
                all_models.add(model)
                all_tasks.add(task)
                data[task] = data.get(task, {})
                data[task][model] = data[task].get(model, {})
                data[task][model][size] = data[task][model].get(size, []) + [float(acc)]
                key = (task, model, size)
                items[key] = items.get(key, []) + [float(acc)]
    items = {k : numpy.array(v) for k, v in items.iteritems()}
    vals = [(t, m, int(s), v.mean(), v.mean() + v.std(), v.mean() - v.std()) for (t, m, s), v in items.iteritems()]
    names = ["task", "model", "size", "score", "plus", "minus"]
    df = ro.DataFrame({
        "task" : ro.StrVector([x[0] for x in vals]),
        "model" : ro.StrVector([x[1] for x in vals]),
        "size" : ro.IntVector([x[2] for x in vals]),
        "score" : ro.FloatVector([x[3] for x in vals]),
        "plus" : ro.FloatVector([x[4] for x in vals]),
        "minus" : ro.FloatVector([x[5] for x in vals]),
    })
    grdevices.png(file=options.output, width=512, height=512)
    x = ggplot2.ggplot(df) + ggplot2.aes_string(x="size", y="score", color="model", ymin="minus", ymax="plus") + ggplot2.geom_line() + ggplot2.geom_ribbon(alpha=.5) + ggplot2.ggtitle(task.title()) + ggplot2.labs(y="F-Score" if task == "events" else "Accuracy", x="Document Count") + ggplot2.theme_bw() + ggplot2.theme(text=ggplot2.element_text(size=20, family="serif"), legend_title=ggplot2.element_blank(), plot_title=ggplot2.element_text(hjust=.5, size=20, family="serif"))
    x.plot()
    grdevices.dev_off()
