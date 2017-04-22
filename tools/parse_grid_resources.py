import re
import gzip
import pickle
import subprocess


pattern = re.compile(r""".*
ru_wallclock\s+(?P<wallclock>[^\n]+)\s+
ru_utime\s+(?P<usertime>[^\n]+)\s+
ru_stime\s+(?P<systemtime>[^\n]+)\s+
ru_maxrss\s+(?P<maxresidentset>[^\n]+)\s+
ru_ixrss\s+([^\n]+)\s+
ru_ismrss\s+([^\n]+)\s+
ru_idrss\s+([^\n]+)\s+
ru_isrss\s+([^\n]+)\s+
ru_minflt\s+([^\n]+)\s+
ru_majflt\s+([^\n]+)\s+
ru_nswap\s+([^\n]+)\s+
ru_inblock\s+([^\n]+)\s+
ru_oublock\s+([^\n]+)\s+
ru_msgsnd\s+([^\n]+)\s+
ru_msgrcv\s+([^\n]+)\s+
ru_nsignals\s+([^\n]+)\s+
ru_nvcsw\s+([^\n]+)\s+
ru_nivcsw\s+([^\n]+)\s+
wallclock\s+([^\n]+)\s+
cpu\s+([^\n]+)\s+
mem\s+([^\n]+)\s+
io\s+([^\n]+)\s+
iow\s+([^\n]+)\s+
ioops\s+([^\n]+)\s+
maxvmem\s+([^\n]+)\s+
maxrss\s+([^\n]+)\s+
maxpss\s+([^\n]+)\s+
.*
""", re.DOTALL | re.VERBOSE)


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--id", dest="id", type=int, default=0)
    parser.add_argument("-o", "--output", dest="output")
    options = parser.parse_args()    

    p = subprocess.Popen(["qacct", "-j", "%d" % (options.id)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    
    resources = {k : float(v) for k, v in pattern.match(err + out).groupdict().iteritems()}
    
    with gzip.open(options.output, "w") as ofd:
        pickle.dump(resources, ofd)
