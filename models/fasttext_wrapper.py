import gzip
import pickle
import codecs
import logging
import tempfile
from itertools import chain
import subprocess
import os
import shlex
import shutil
import math
import re

writer = codecs.getwriter("utf-8")
reader = codecs.getreader("utf-8")

if __name__ == "__main__":
    
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input")
    parser.add_argument("--train", dest="train")
    parser.add_argument("--test", dest="test")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--output", dest="output")
    
    parser.add_argument("--min_char_ngram", dest="min_char_ngram", type=int, default=0)
    parser.add_argument("--max_char_ngram", dest="max_char_ngram", type=int, default=4)
    
    parser.add_argument("--word_vector_size", dest="word_vector_size", type=int, default=100)
    parser.add_argument("--word_context_size", dest="word_context_size", type=int, default=5)
    parser.add_argument("--epochs", dest="epochs", type=int, default=100)
    parser.add_argument("--negatives_sampled", dest="negatives_sampled", type=int, default=5)
    parser.add_argument("--min_word_count", dest="min_word_count", type=int, default=0)
    parser.add_argument("--min_label_count", dest="min_label_count", type=int, default=0)
    parser.add_argument("--max_word_ngram", dest="max_word_ngram", type=int, default=1)
    parser.add_argument("--bucket_count", dest="bucket_count", type=int, default=2000000)
    parser.add_argument("--threads", dest="threads", type=int, default=1)
    parser.add_argument("--verbosity", dest="verbosity", type=int, default=1)
    
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.1)
    parser.add_argument("--learning_rate_update_rate", dest="learning_rate_update_rate", type=int, default=100)
    parser.add_argument("--sampling_threshold", dest="sampling_threshold", type=float, default=0.0001)
    
    parser.add_argument("--label_prefix", dest="label_prefix", default="__label__")
    parser.add_argument("--loss_function", dest="loss_function", choices=["ns", "hs", "softmax"], default="ns")
    options = parser.parse_args()

    args = {k : v for k, v in options._get_kwargs()}
    train_command = "fasttext supervised -input %(input_file)s -output %(output_file)s -lr %(learning_rate)f -lrUpdateRate %(learning_rate_update_rate)d -dim %(word_vector_size)d -ws %(word_context_size)d -epoch %(epochs)d -minCount %(min_word_count)d -minCountLabel %(min_label_count)d -neg %(negatives_sampled)d -wordNgrams %(max_word_ngram)d -loss %(loss_function)s -bucket %(bucket_count)d -minn %(min_char_ngram)d -maxn %(max_char_ngram)d -thread %(threads)d -t %(sampling_threshold)f -label %(label_prefix)s -verbose %(verbosity)d"
    apply_command = "fasttext predict-prob %(model)s %(test_file)s %(num_labels)d"

    # training
    if options.train and options.output and options.input:
        with reader(gzip.open(options.train)) as ifd:
            indices = [int(l.strip()) for l in ifd]
        indices = set(indices)
        instances, labels = [], []        
        with reader(gzip.open(options.input)) as ifd:
            for i, line in enumerate(ifd):
                if i in indices:
                    cid, label, text = line.strip().split("\t")
                    instances.append(text)
                    labels.append(label)
        _, model_fname = tempfile.mkstemp(suffix=".bin")
        base_fname = os.path.splitext(model_fname)[0]
        vec_fname = "%s.vec" % (base_fname)
        _, input_fname = tempfile.mkstemp()
        label_lookup = {}
        for l in labels:
            label_lookup[l] = label_lookup.get(l, len(label_lookup))
        with writer(open(input_fname, "w")) as ofd:
            for label, text in zip(labels, instances):
                ofd.write("__label__%s %s\n" % (label, text))
        logging.info("Training with %d instances, %d labels", len(instances), len(label_lookup))
        args["input_file"] = input_fname
        args["output_file"] = base_fname
        toks = shlex.split(train_command % args)
        p = subprocess.Popen(toks)
        p.communicate()
        os.remove(input_fname)
        shutil.copyfile(model_fname, options.output)
        os.remove(model_fname)

    # testing
    elif options.test and options.model and options.output and options.input:
        with reader(gzip.open(options.test)) as ifd:
            indices = [int(l.strip()) for l in ifd]
        indices = set(indices)
        instances, gold = [], []
        labels = set()
        _, test_fname = tempfile.mkstemp()
        with reader(gzip.open(options.input)) as ifd, writer(open(test_fname, "w")) as ofd:
            for i, line in enumerate(ifd):
                if i in indices:
                    cid, label, text = line.strip().split("\t")
                    instances.append(text)
                    gold.append((cid, label))
                    ofd.write(text + "\n")
                    labels.add(label)
        
        logging.info("Testing with %d instances", len(instances))
        args["num_labels"] = 100
        args["test_file"] = test_fname
        p = subprocess.Popen(shlex.split(apply_command % args), stdout=subprocess.PIPE)
        out, err = p.communicate()
        os.remove(test_fname)
        
        data = {}
        for line, (cid, label) in zip(out.strip().split("\n"), gold):
            probs = {k : math.log(float(v)) for k, v in [m.groups() for m in re.finditer(r"__label__(\S+) (\S+)", line)]}
            for k in probs.keys():
                labels.add(k)
            data[cid] = (label, probs)
        codes = sorted(labels)
        with writer(gzip.open(options.output, "w")) as ofd:
            ofd.write("\t".join(["DOC", "GOLD"] + codes) + "\n")
            for cid, (label, probs) in data.iteritems():
                ofd.write("\t".join([cid, label] + [str(probs.get(c, float("-inf"))) for c in codes]) + "\n")
    else:
        print "ERROR: you must specify --input and --output, and either --train or --test and --model!"
