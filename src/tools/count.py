import gzip
from io import get_count, writer

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input")
    parser.add_argument("-o", "--output", dest="output")
    options = parser.parse_args()

    n = get_count(options.input)
    with writer(gzip.open(options.output, "w")) as ofd:
        ofd.write(str(n))
