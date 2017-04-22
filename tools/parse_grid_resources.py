import re
import gzip
import pickle
import subprocess


pattern = re.compile(r""".*
ru_wallclock\s+(?P<wallclock>\S+)\s+
ru_utime\s+(?P<usertime>\S+)\s+
ru_stime\s+(?P<systemtime>\S+)\s+
ru_maxrss\s+(?P<maxresidentset>\S+)\s+
ru_ixrss\s+(\S+)\s+
ru_ismrss\s+(\S+)\s+
ru_idrss\s+(\S+)\s+
ru_isrss\s+(\S+)\s+
ru_minflt\s+(\S+)\s+
ru_majflt\s+(\S+)\s+
ru_nswap\s+(\S+)\s+
ru_inblock\s+(\S+)\s+
ru_oublock\s+(\S+)\s+
ru_msgsnd\s+(\S+)\s+
ru_msgrcv\s+(\S+)\s+
ru_nsignals\s+(\S+)\s+
ru_nvcsw\s+(\S+)\s+
ru_nivcsw\s+(\S+)\s+
wallclock\s+(\S+)\s+
cpu\s+(\S+)\s+
mem\s+(\S+)\s+
io\s+(\S+)\s+
iow\s+(\S+)\s+
ioops\s+(\S+)\s+
maxvmem\s+(\S+)\s+
maxrss\s+(\S+)\s+
maxpss\s+(\S+)\s+
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
    
    resources = pattern.match(err).groupdict()
    
    with gzip.open(options.output, "w") as ofd:
        pickle.dump(resources, ofd)
