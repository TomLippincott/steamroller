import gzip
import codecs

writer = codecs.getwriter("utf-8")
reader = codecs.getreader("utf-8")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input")
    parser.add_argument("-o", "--output", dest="output")
    options = parser.parse_args()

    counter = 0
    with reader(gzip.open(options.input)) as ifd:
        for line in ifd:
            counter += 1
    with writer(gzip.open(options.output, "w")) as ofd:
        ofd.write(str(counter))
