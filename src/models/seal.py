#!/usr/bin/env python



#
# Data-loading functions (can probably skip this)
#


def instances_and_lookups(input_file, index_file, sym_lookup={"unk" : 0}, label_lookup={"unk" : 0}):
    """
    Read communications and create integer encodings for them, along with lookups to recover the
    strings.  "unk" is mapped to 0 for both symbols and labels, to handle OOV at test time.  If
    symbol or label lookups are passed to the function, does *not* update the lookups and encodes
    unseen items as "unk".

    In other words, when reading training data, don't pass lookups.  When reading test data, pass in
    the lookups from the training data.
    """
    assert(sym_lookup["unk"] == 0 and label_lookup["unk"] == 0)
    update_sym = len(sym_lookup) == 1
    update_label = len(label_lookup) == 1
    cid_lookup = {}
    instances, labels = [], []
    unk_sym_occs, unk_sym_types = 0, set()
    unk_label_occs, unk_label_types = 0, set()
    for cid, label, text in read_data(options.input, options.train if options.train else options.test):
        if update_label:
            label_lookup[label] = label_lookup.get(label, len(label_lookup))
        cid_lookup[cid] = label_lookup.get(cid, len(cid_lookup))
        syms = []
        for c in text:
            if update_sym:
                sym_lookup[c] = sym_lookup.get(c, len(sym_lookup))
            syms.append(sym_lookup.get(c, 0))
            if syms[-1] == 0:
                unk_sym_types.add(c)
                unk_sym_occs += 1
        instances.append((label_lookup.get(label, 0), syms, cid_lookup[cid]))
        if instances[-1][0] == 0:
            unk_sym_types.add(label)
            unk_label_occs += 1            
    logging.info("Loaded %d instances, %d labels", len(instances), len(label_lookup))
    logging.info("%d/%d unknown symbol occurrences/types, %d/%d unknown label occurences/types",
                 unk_sym_occs,
                 len(unk_sym_types),
                 unk_label_occs,
                 len(unk_label_types),                 
    )
    return instances, cid_lookup, sym_lookup, label_lookup


def example_generator(examples):
    """
    Simply yields examples per TF idiom.
    """
    for example in examples:
        yield {'text': example[1], 'label': example[0], "cid" : example[2]}


def serialize_model(sess, temppath, sym_lookup, label_lookup, cid_lookup, output_file):
    with gzip.open(os.path.join(temppath, "lookups.pkl.gz"), "w") as ofd:
        pickle.dump((sym_lookup, label_lookup, cid_lookup), ofd)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(temppath, "model"))
    #with open(os.path.join(checkpoint_path, "checkpoint")) as ifd:
    #    text = ifd.read()
    #    text = re.sub(checkpoint_path, "TMPPATH", text)
    #with open(os.path.join(checkpoint_path, "checkpoint"), "w") as ofd:
    #    ofd.write(text)
    with tarfile.open(output_file, "w:gz") as ofd:
        for name in glob(os.path.join(temppath, "*")):
            ofd.add(name, arcname=os.path.basename(name))


def deserialize_model(model_file, output_path):
    with tarfile.open(model_file) as ifd:
        ifd.extractall(path=output_path)
    #with open(os.path.join(output_path, "checkpoint")) as ifd:
    #    text = ifd.read()
    #    text = re.sub("TMPPATH", temppath, text)
    #with open(os.path.join(output_path, "checkpoint"), "w") as ofd:
    #    ofd.write(text)
    with gzip.open(os.path.join(output_path, "lookups.pkl.gz")) as ifd:
        sym_lookup, label_lookup, _ = pickle.load(ifd)
    return sym_lookup, label_lookup


#
# Callbacks for inference mode (irrelevant to training)
#


def key_callback(id_to_label, id_to_cid, x):
    """
    When TF iterates over examples, map the labels and IDs from integers back to strings.
    """
    return (id_to_label[x["label"]], id_to_cid[x["cid"]])


def result_callback(id_to_label, output_file, res):
    """
    When TF inference completes, assemble the results to write to disk.
    """
    data = {}
    order = [id_to_label[i] for i in range(len(id_to_label))]
    correct, total = 0, 0
    for (gold, cid), (probs,) in res:
        total += 1
        if gold == order[probs.flatten().argmax()]:
            correct += 1
        data[cid] = (gold, {k : v for k, v in zip(order, probs.flatten())})
    logging.info("Accuracy: %.3f", float(correct) / total)
    write_probabilities(data, options.output)

        
#
# Modified Seal code
#
# General goals: remove global variables as much as possible, and annotate
#                code with what I believe it does.  Remove and consolidate
#                things to a minimal functional flat RNN.
#

class SequenceModel(object):

    def __init__(self, num_letters, num_labels, char_rnn_layer, char_state_vector_length, char_embed_vector_length):
        logging.info("Creating model with %d letters and %d labels", num_letters, num_labels)
        #self._keep_probability = tf.Variable(1.0, trainable=False)
        self._keep_probability = tf.placeholder_with_default(1.0, [], name='keep_probability')
        self._char_cells = []
        for l in range(char_rnn_layer):
            char_cell = tf.contrib.rnn.GRUCell(num_units=char_state_vector_length)
            char_cell = tf.contrib.rnn.DropoutWrapper(char_cell,
                                                      input_keep_prob=self._keep_probability,
                                                      output_keep_prob=self._keep_probability)
            char_cell = td.ScopedLayer(char_cell, 'char_cell_{}'.format(l))
            self._char_cells.append(char_cell)

        # Character-level stacked RNN
        self._char_embed = (td.Map(td.Scalar(tf.int64) >>
                             td.Function(td.Embedding(num_letters,
                                                      char_embed_vector_length,
                                                      name='char_embed'))))
        self._char_rnn = self._char_embed >> td.RNN(self._char_cells[0])
        for l in range(1, char_rnn_layer):
            self._char_rnn >>= td.GetItem(0) >> td.RNN(self._char_cells[l])

        self._context = self._char_rnn >> td.GetItem(1)    

        self._output_layer = self._context >> td.FC(num_labels,
                                                    activation=None,
                                                    input_keep_prob=self._keep_probability,
                                                    name='output_layer')

        self._label = td.Scalar(tf.int64)
        self._root_block = td.Record([('text', self._output_layer),
                                      ('label', self._label)])

        self._compiler = td.Compiler.create(self._root_block)
        (self._logits, self._labels) = self._compiler.output_tensors

        self._loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._logits,
            labels=self._labels
        )
        self._global_step = tf.contrib.framework.get_or_create_global_step()
        lr = tf.train.exponential_decay(.001, self._global_step,
                                    100000, 0.96, staircase=True)

        self._optimizer = build_optimizer_from_params("adam", learning_rate=.001)
        tvars = tf.trainable_variables()

        self._predictions = tf.argmax(self._logits, 1)
        self._accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self._predictions, self._labels), dtype=tf.float32)
        )

        self._grads_and_vars = self._optimizer.compute_gradients(self._loss, tvars)

        tf.clip_by_global_norm([x[0] for x in self._grads_and_vars], 10.0)
        self._train_op = self._optimizer.minimize(self._loss, global_step=self._global_step)

    @property
    def loss(self):
        return self._loss

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def compiler(self):
        return self._compiler

    @property
    def train_op(self):
        return self._train_op

    @property
    def global_step(self):
        return self._global_step

    @property
    def predictions(self):
        return self._predictions

    @property
    def labels(self):
        return self._labels

    @property
    def log_probs(self):
        return self._logits
    
    @property
    def outputs(self):
        return [self._logits]
    
    def build_feed_dict(self, expressions):
        return self._compiler.build_feed_dict(expressions)


#
# Entrypoint for script
#


if __name__ == "__main__":
    import re
    import gzip
    import pickle
    import codecs
    import logging
    from functools import partial
    from itertools import chain
    import random
    import tempfile
    import shutil
    import os.path
    import tarfile
    from glob import glob
    import tensorflow as tf
    from tensorflow.python.ops import math_ops, array_ops
    from tensorflow.python.ops import variable_scope as vs
    import tensorflow.contrib.seq2seq as seq2seq
    import tensorflow_fold as td
    import tensorflow_fold.blocks.blocks
    import tensorflow_fold.blocks.result_types as tdt
    from tensorflow_fold.blocks.plan import build_optimizer_from_params
    from tqdm import tqdm
    import numpy
    from steamroller.tools.io import read_data, write_probabilities, writer, reader, extract_character_ngrams

    import argparse

    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input")
    parser.add_argument("--train", dest="train")
    parser.add_argument("--test", dest="test")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--output", dest="output")
    
    parser.add_argument("--dev", dest="dev", type=float, default=.1)
    parser.add_argument("--char_embed_vector_length", dest="char_embed_vector_length", default=512, type=int)
    parser.add_argument("--char_state_vector_length", dest="char_state_vector_length", default=512, type=int)
    parser.add_argument("--char_rnn_layer", dest="char_rnn_layer", default=2, type=int)
    parser.add_argument("--keep_probability", dest="keep_probability", default=.5, type=float)
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", default=10.0, type=float)
    parser.add_argument("--learning_rate", dest="learning_rate", default=.001, type=float)

    parser.add_argument("--epochs", dest="epochs", default=100, type=int)
    parser.add_argument("--batch_size", dest="batch_size", default=32, type=int)
    parser.add_argument("--batches_per_epoch", dest="batches_per_epoch", default=100, type=int)
    options = parser.parse_args()
    
    temppath = tempfile.mkdtemp(prefix="tlippincott-tf")

    try:

        # train model
        if options.train and options.output and options.input:
            instances, cid_lookup, sym_lookup, label_lookup = instances_and_lookups(options.input,
                                                                                    options.train)
            random.shuffle(instances)
            train = instances[0 : int(len(instances) * (1.0 - options.dev))]
            dev = instances[int(len(instances) * (1.0 - options.dev)):]
            best_score = None
            with tf.Graph().as_default():
                model = SequenceModel(len(sym_lookup),
                                      len(label_lookup),
                                      options.char_rnn_layer,
                                      options.char_state_vector_length,
                                      options.char_embed_vector_length,
                )
                #init = tf.assign(model._keep_probability, options.keep_probability)
                init = tf.global_variables_initializer()
                #supervisor = tf.train.Supervisor(logdir=temppath)
                #with supervisor.managed_session() as sess:
                with tf.Session() as sess:
                    sess.run(init)
                    since_last_write = 0
                    #sess.run(init) #tf.assign(model._keep_probability, options.keep_probability))
                    for e in xrange(options.epochs):
                        logging.info("Epoch %d", e + 1)
                        random.shuffle(train)
                        accs = []
                        for i, batch_feed in enumerate(model.compiler.build_loom_input_batched(example_generator(train), options.batch_size)):
                            if i > options.batches_per_epoch:
                                break
                            else:
                                _, step, loss_v, acc_v = sess.run(
                                    [model.train_op, model.global_step, model.loss, model.accuracy],
                                    feed_dict={model.compiler.loom_input_tensor : batch_feed, model._keep_probability : options.keep_probability})
                                accs.append(acc_v)
                        logging.info("Train accuracy over epoch #%d was %f", e + 1, sum(accs) / len(accs))
                        accs = []
                        for batch_feed in model.compiler.build_loom_input_batched(example_generator(dev), options.batch_size):
                            acc, preds, prob = sess.run([model.accuracy, model.predictions, model.log_probs],
                                                        feed_dict={model.compiler.loom_input_tensor : batch_feed, model._keep_probability : 1.0})
                            accs.append(acc)
                            #probs.append(prob)
                        acc = sum(accs) / len(accs)
                        logging.info("Dev accuracy over epoch #%d was %f", e + 1, acc)
                        since_last_write += 1
                        if since_last_write > 5:
                            logging.info("Stopping!")
                            break
                        if best_score == None or acc > best_score:
                            logging.info("New best score: %f", acc)
                            best_score = acc
                            since_last_write = 0
                            serialize_model(sess, temppath, sym_lookup, label_lookup, cid_lookup, options.output)
            
        # perform inference with existing model
        elif options.test and options.model and options.output and options.input:
            train_sym_lookup, train_label_lookup = deserialize_model(options.model, temppath)
            instances, cid_lookup, sym_lookup, label_lookup = instances_and_lookups(options.input,
                                                                                    options.train,
                                                                                    sym_lookup=train_sym_lookup,
                                                                                    label_lookup=train_label_lookup)
            
            random.shuffle(instances)
            id_to_label = {v : k for k, v in label_lookup.iteritems()}
            id_to_cid = {v : k for k, v in cid_lookup.iteritems()}
            test = example_generator(instances)
            
            with tf.Graph().as_default():
                model = SequenceModel(len(sym_lookup),
                                      len(label_lookup),
                                      options.char_rnn_layer,
                                      options.char_state_vector_length,
                                      options.char_embed_vector_length,
                )
                #init = tf.global_variables_initializer()
                #init = tf.assign(model._keep_probability, options.keep_probability)
                #supervisor = tf.train.Supervisor(logdir=temppath)
                data = {}
                #with supervisor.managed_session() as sess:
                with tf.Session() as sess:
                    s = tf.train.Saver()
                    s.restore(sess, os.path.join(temppath, "model"))
                    #sess.run(init) #tf.assign(model._keep_probability, 1.0))
                    accs = []
                    probs = []
                    for batch_feed in model.compiler.build_loom_input_batched(test, options.batch_size):
                        
                        acc, preds, prob = sess.run([model.accuracy, model.predictions, model.log_probs],
                                                     feed_dict={model.compiler.loom_input_tensor : batch_feed, model._keep_probability : 1.0})
                        accs.append(acc)
                        probs.append(prob)
                probs = numpy.concatenate(probs, axis=0)
                logging.info("Accuracy: %f", sum(accs) / len(accs))
                data = {}
                order = [id_to_label[i] for i in range(len(id_to_label))]
                for dist, (lid, _, cid) in zip(probs, instances):
                    cid = id_to_cid.get(cid, str(cid))
                    g = id_to_label[lid]
                    data[cid] = (g, {k : v for k, v in zip(order, dist.flatten())})

                write_probabilities(data, options.output)
        else:
            logging.error("You must specify the input and output files, and either a training file, or testing and model files!")
    finally:
        shutil.rmtree(temppath)
