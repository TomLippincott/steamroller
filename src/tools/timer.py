import argparse
import resource
import subprocess
import pickle
import gzip

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="output", help="Output file")
    parser.add_argument(nargs="+", dest="command", help="Command")
    args = parser.parse_args()

    p = subprocess.Popen(args.command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print(err)
        raise Exception()
    u = resource.getrusage(resource.RUSAGE_SELF)

    with gzip.open(args.output, "w") as ofd:
        pickle.dump(u, ofd)
