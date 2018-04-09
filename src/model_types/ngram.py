#!/usr/bin/env python

import gzip
import codecs
from itertools import chain
import pickle
import logging
import math
import numpy
from steamroller.tools.io import read_data, write_probabilities, writer, reader
from nltk.util import ngrams

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input")
    parser.add_argument("--train", dest="train")
    parser.add_argument("--tag-type", dest="tag_type", default="attribute")
    parser.add_argument("--dev", dest="dev")
    parser.add_argument("--test", dest="test")
    parser.add_argument("--prior", dest="prior", default=False, action="store_true")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--output", dest="output")
    parser.add_argument("--max_ngram", dest="max_ngram", type=int, default=4)
    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    left, right = ("<L>", "<R>")
    
    if options.train and options.output and options.input:
        counts = {}
        model = {}
        data = {}
        #backoffs = {i : 1.0 / options.max_ngram for i in range(1, options.max_ngram + 1)}
        for cid, label, text in read_data(options.input, options.train, tag_type=options.tag_type):
            model[label] = model.get(label, {})
            model[label][0] = model[label].get(0, 0) + 1
            textB = [left for i in range(options.max_ngram)] + list(text) + [right for i in range(options.max_ngram)]
            for n in range(1, options.max_ngram + 1):
                model[label][n] = model[label].get(n, {})
                #for g in ngrams(text, n, pad_left=True, pad_right=True, left_pad_symbol=left, right_pad_symbol=right):
                for g in ngrams(text, n): 
                    model[label][n][g] = model[label][n].get(g, 0) + 1
            #model[label] = model.get(label, {i : {} for i in range(options.max_ngram + 1)})
            #counts[label] = counts.get(label, 0) + 1
            #seq = [None for i in range(options.max_ngram)]
            #dummy = tuple([None for i in range(options.max_ngram - 1)])
            #model[label][options.max_ngram - 1][dummy] = model[label][options.max_ngram - 1].get(dummy, 0) + 1 
            #for c in list(text) + [None]:
            #    seq = seq[1:]
            #    seq.append(c)
            #    for n in range(options.max_ngram + 1):
            #        gram = tuple(seq[-n:]) if n > 0 else ()
            #        model[label][n][gram] = model[label][n].get(gram, 0) + 1

        with gzip.open(options.output, "w") as ofd:
            pickle.dump(model, ofd)
            
    # testing
    elif options.test and options.model and options.output and options.input:
        instances, gold = [], []
        with gzip.open(options.model) as ifd:
            model = pickle.load(ifd)

        scores = {}
        max_n = max(list(model.values())[0].keys())
        totals = {k : sum(v[1].values()) for k, v in model.items()}
        for cid, label, text in read_data(options.input, options.test, tag_type=options.tag_type):
            probs = {k : 0.0 for k in model.keys()}
            for g in ngrams(text, max_n, pad_left=True, pad_right=True, left_pad_symbol=left, right_pad_symbol=right):
                for l in model.keys():
                    gp = g
                    while len(gp) >= 0:
                        den = gp[:-1]
                        if len(gp) == 0:
                            probs[l] += math.log(0.000000001)
                            break
                        elif gp in model[l][len(gp)]:
                            num = model[l][len(gp)][gp]
                            denom = (totals[l] if len(gp) == 1 else model[l][len(gp) - 1][den])
                            prob = model[l][len(gp)][gp] / float(totals[l] if len(gp) == 1 else model[l][len(gp) - 1][den])
                            assert(prob > 0)
                            assert(prob <= 1)
                            probs[l] += math.log(prob)
                            break
                        else:
                            gp = gp[1:]
                            
            scores[cid] = (label, probs)

            #n in reversed(range(1, max_n + 1)):
                
            #    print(n)
            #instances.append(text)
            #gold.append((cid, label))



            
        # total = sum(counts.values())
        # prior = {k : math.log(v / float(total)) for k, v in counts.items()}
        # labels = model.keys()
        # logging.info("Testing with %d instances, %d labels", len(instances), len(labels))
        # # data = {}
        # #backoffs = {i : 1.0 / options.max_ngram for i in range(1, options.max_ngram + 1)}
        # for (cid, label), text in zip(gold, instances):
        #     total_probs = {k : v for k, v in prior.items()}
        #     seq = [None for i in range(options.max_ngram)]
        #     for c in list(text) + [None]:
        #         seq = seq[1:]
        #         seq.append(c)
        #         single_probs = {}
        #         for n in range(1, options.max_ngram + 1):
        #             b = backoffs[n]
        #             num_gram = tuple(seq[-n:])
        #             den_gram = num_gram[:-1]
        #             for l, v in model.items():
        #                 num = v[n].get(num_gram, 0)
        #                 #print(num_gram, den_gram, l)                        
        #                 if num > 0:

        #                     den = v[n - 1][den_gram]                            
        #                     #print(num, den)
        #                     prob = num / den
        #                     #print(prob, b)
        #                     single_probs[l] = single_probs.get(l, 0.0) + (prob * b)                
        #         for l, p in single_probs.items():
        #             if p != 0.0:
        #                 total_probs[l] += math.log(p)
        #     scores[cid] = (label, total_probs)
        #                     #print(prob)
        #             #prob += model[n][gram] * b
        #             #model[n][gram] = model[n].get(gram, 0) + 1
            
        write_probabilities(scores, options.output)
        
