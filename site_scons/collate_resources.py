#!/usr/bin/env python

import gzip
import re
import math
import csv

pattern = re.compile(r"""
.*User\ time\ \(seconds\):\ (?P<usertime>\S+)\s+
System\ time\ \(seconds\):\ (?P<systemtime>\S+)\s+
Percent\ of\ CPU\ this\ job\ got:\ (?P<cpupercent>\S+)%\s+
Elapsed\ \(wall\ clock\)\ time\ \(h:mm:ss\ or\ m:ss\):\ (\S+)\s+
Average\ shared\ text\ size\ \(kbytes\):\ (\S+)\s+
Average\ unshared\ data\ size\ \(kbytes\):\ (\S+)\s+
Average\ stack\ size\ \(kbytes\):\ (?P<avgstack>\S+)\s+
Average\ total\ size\ \(kbytes\):\ (?P<avgtotal>\S+)\s+
Maximum\ resident\ set\ size\ \(kbytes\):\ (?P<maxresidentset>\S+)\s+
Average\ resident\ set\ size\ \(kbytes\):\ (?P<avgresidentset>\S+)\s+
Major\ \(requiring\ I/O\)\ page\ faults:\ (\S+)\s+
Minor\ \(reclaiming\ a\ frame\)\ page\ faults:\ (\S+)\s+
Voluntary\ context\ switches:\ (\S+)\s+
Involuntary\ context\ switches:\ (\S+)\s+
Swaps:\ (\S+)\s+
File\ system\ inputs:\ (\S+)\s+
File\ system\ outputs:\ (\S+)\s+
Socket\ messages\ sent:\ (\S+)\s+
Socket\ messages\ received:\ (\S+)\s+
Signals\ delivered:\ (\S+)\s+
Page\ size\ \(bytes\):\ (\S+)\s+
Exit\ status:\ (\S+).*""", re.DOTALL | re.VERBOSE)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument(nargs="+", dest="inputs")
    options = parser.parse_args()
    
    vals = {}
    for f in options.inputs:
        task, model, size, fold, _ = re.match(r"^work/(.*?)_(.*?)_(.*?)_(.*?)(\.|_).*txt$", f).groups()
        size = int(size)
        fold = int(fold)
        with open(f) as ifd:
            key = (task, size, model, fold)
            res = {k : float(v) for k, v in pattern.search(ifd.read()).groupdict().iteritems()}
            vals[key] = (res["maxresidentset"], res["usertime"])

    with gzip.open(options.output, "w") as ofd:
        c = csv.DictWriter(ofd, fieldnames=["task", "size", "model", "fold", "Memory", "CPU"], delimiter="\t")
        c.writeheader()
        for (task, size, model, fold), (mem, cpu) in sorted(vals.iteritems()):
            c.writerow({"task" : task, "model" : model, "size" : size, "fold" : fold, "Memory" : mem / 1000000.0, "CPU" : cpu})

