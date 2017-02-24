import gzip
import codecs
import random

writer = codecs.getwriter("utf-8")
reader = codecs.getreader("utf-8")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input")
    parser.add_argument("--total_file", dest="total_file")
    parser.add_argument("--size", dest="size", type=int)
    parser.add_argument("--proportion", dest="proportion", type=float)    
    parser.add_argument("--train", dest="train")
    parser.add_argument("--test", dest="test")
    options = parser.parse_args()

    with reader(gzip.open(options.total_file)) as ifd:
        total = int(ifd.read().strip())

    size = options.size
    train_proportion = options.proportion
    train_count = int(size * train_proportion)
    indices = range(total)
    random.shuffle(indices)
    
    with writer(gzip.open(options.train, "w")) as ofd:
        ofd.write("\n".join([str(i) for i in indices[0:train_count]]))

    with writer(gzip.open(options.test, "w")) as ofd:
        ofd.write("\n".join([str(i) for i in indices[train_count:size]]))


