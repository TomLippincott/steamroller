import gzip
import pickle
import codecs
import logging
from itertools import chain
import random
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
import tensorflow_fold as td
import tensorflow_fold.blocks.blocks
import tensorflow_fold.blocks.result_types as tdt
from tensorflow_fold.blocks.plan import build_optimizer_from_params
import tensorflow.contrib.seq2seq as seq2seq
from seal import data
from tqdm import tqdm


writer = codecs.getwriter("utf-8")
reader = codecs.getreader("utf-8")


_OPTIMIZER_CLASSES = {
    'adadelta': tf.train.AdadeltaOptimizer,
    'adagradda': tf.train.AdagradDAOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'adam': tf.train.AdamOptimizer,
    'ftrl': tf.train.FtrlOptimizer,
    'gradientdescent': tf.train.GradientDescentOptimizer,
    'momentum': tf.train.MomentumOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
}


def get_cell_fn(cell_type, cell_args):
    cell_fn = None
    if cell_type == 'gru':
        cell_fn = tf.contrib.rnn.GRUCell
    elif cell_type == 'lstm':
        cell_fn = tf.contrib.rnn.LSTMCell
    elif cell_type == 'gridlstm':
        cell_fn = tf.contrib.grid_rnn.Grid2LSTMCell
        cell_args['use_peepholes'] = True
        cell_args['forget_bias'] = 1.0
    elif cell_type == 'gridgru':
        cell_fn = tf.contrib.grid_rnn.Grid2GRUCell
    else:
        raise Exception("unsupported cell type: {}".format(cell_type))
    return cell_fn


    class AttentionLayer(td.TensorToTensorLayer):
        def __init__(self, num_units_out, name=None):
            self._activation = tf.tanh
            self._initializer = tf.uniform_unit_scaling_initializer(1.15)
            self._num_units_out = num_units_out
            if name is None: name = 'AttentionLayer_%d' % num_units_out
            super(AttentionLayer, self).__init__(
                output_type=tdt.TensorType([num_units_out]), name_or_scope=name)

        @property
        def output_size(self):
            return self.output_type.shape[0]

        def _create_variables(self):
            if self.input_type.dtype != 'float32':
                raise TypeError('AttentionLayer input dtype must be float32: %s' %
                                self.input_type.dtype)
            if self.input_type.ndim != 1:
                raise TypeError('AttentionLayer input shape must be 1D: %s' %
                                str(self.input_type.shape))
            self._attn_v = tf.get_variable(
                'attn_v', [self.output_type.shape[0]],
                initializer=tf.truncated_normal_initializer(stddev=0.01))
            self._attn_W = tf.get_variable(
                'attn_W', [self.input_type.shape[0], self.output_type.shape[0]],
                initializer=self._initializer)

        def _process_batch(self, batch):
            return self._attn_v * self._activation(tf.matmul(batch, self._attn_W))

    def Attention(num_units, name=None):
        attention = td.Composition(name=name)
        with attention.scope():
            a = td.Function(AttentionLayer(num_units))
            h = attention.input
            exp_e = td.Map(a >> td.Function(tf.exp)).reads(h)
            z = (td.Sum() >> td.Broadcast()).reads(exp_e)
            alpha = td.ZipWith(td.Function(tf.div)).reads(exp_e, z)
            c = (td.ZipWith(td.Function(tf.multiply)) >> td.Sum()).reads(alpha, h)
            attention.output.reads(c)
        return attention.set_constructor_name('Attention')


# Either attention or last state
def sequence_to_tensor(rnn_output, num_units, attention):
    if attention == 'last':
        print('[sequence to tensor] last state')
        final_state = rnn_output >> td.GetItem(1)
        return final_state
    elif attention == 'mixture':
        print('[sequence to tensor] attention')
        rnn_states = rnn_output >> td.GetItem(0)
        return rnn_states >> Attention(num_units)


def create_flat_model(num_letters, num_labels, args):
    print('{} letters {} labels'.format(num_letters, num_labels))

    # Create a placeholder for dropout, if we are in train mode.
    #keep_prob = (tf.placeholder_with_default(1.0, [], name='keep_prob')
    #             if plan.mode == plan.mode_keys.TRAIN else None)
    keep_prob = True

    char_cells = []
    for l in range(args.char_rnn_layer):
        cell_args = {'num_units': args.char_state_vector_length}
        cell_fn = get_cell_fn("gru", cell_args) #FLAGS.char_cell_type, cell_args)
        char_cell = cell_fn(**cell_args)
        char_cell = tf.contrib.rnn.DropoutWrapper(char_cell,
                                                  input_keep_prob=keep_prob,
                                                  output_keep_prob=keep_prob)
        char_cell = td.ScopedLayer(char_cell, 'char_cell_{}'.format(l))
        char_cells.append(char_cell)

    # Character-level stacked RNN
    char_embed = (td.Map(td.Scalar(tf.int64) >>
                         td.Function(td.Embedding(num_letters,
                                                  #512,
                                                  args.char_embed_vector_length,
                                                  name='char_embed'))))
    char_rnn = char_embed >> td.RNN(char_cells[0])
    for l in range(1, args.char_rnn_layer):
        char_rnn >>= td.GetItem(0) >> td.RNN(char_cells[l])

    context = sequence_to_tensor(char_rnn,
                                 args.char_state_vector_length,
                                 args.attention
    )
                                 #FLAGS.char_state_vector_length)
    output_layer = context >> td.FC(num_labels,
                                    activation=None,
                                    input_keep_prob=keep_prob,
                                    name='output_layer')

    label = td.Scalar(tf.int64)
    root_block = td.Record([('text', output_layer),
                            ('label', label)])

    # Turn dropout on for training, off for validation.
    #plan.train_feeds[keep_prob] = True #FLAGS.keep_prob

    return root_block


def create_hierarchical_model(num_letters, num_labels, args):
    # Create a placeholder for dropout, if we are in train mode.
    #keep_prob = (tf.placeholder_with_default(1.0, [], name='keep_prob')
    #             if plan.mode == plan.mode_keys.TRAIN else None)
    keep_prob = True

    char_cells = []
    for l in range(args.char_rnn_layer):
        cell_args = {'num_units': args.char_state_vector_length}
        cell_fn = get_cell_fn("gru", args.char_cell_type, cell_args)
        char_cell = cell_fn(**cell_args)
        char_cell = tf.contrib.rnn.DropoutWrapper(char_cell,
                                                  input_keep_prob=keep_prob)
        char_cell = td.ScopedLayer(char_cell, 'char_cell_{}'.format(l))
        char_cells.append(char_cell)

    word_cells = []
    for l in range(args.word_rnn_layer):
        cell_args = {'num_units': args.word_state_vector_length}
        cell_fn = get_cell_fn("gru", args.word_cell_type, cell_args)
        word_cell = cell_fn(**cell_args)
        word_cell = tf.contrib.rnn.DropoutWrapper(word_cell,
                                                  input_keep_prob=keep_prob)
        word_cell = td.ScopedLayer(word_cell, 'word_cell_{}'.format(l))
        word_cells.append(word_cell)

    # Character-level stacked RNN
    char_embed = (td.Map(td.Scalar(tf.int64) >>
                         td.Function(td.Embedding(num_letters,
                                                  #512,
                                                  args.char_embed_vector_length,
                                                  name='char_embed'))))
    char_rnn = char_embed >> td.RNN(char_cells[0])
    for l in range(1, args.char_rnn_layer):
        char_rnn >>= td.GetItem(0) >> td.RNN(char_cells[l])

    # Word-level stacked RNN
    word_embed = td.Map(char_rnn >> td.GetItem(1))
    word_rnn = word_embed >> td.RNN(word_cells[0])
    for l in range(1, args.word_rnn_layer):
        word_rnn >>= td.GetItem(0) >> td.RNN(word_cells[l])

    context = sequence_to_tensor(word_rnn,
                                 args.word_state_vector_length,
                                 #512,
                                 args.attention
    )
                                 #FLAGS.word_state_vector_length)

    output_layer = (context >>
                    td.FC(num_labels,
                          activation=None,
                          input_keep_prob=keep_prob,
                          output_keep_prob=keep_prob,
                          name='output_layer'))

    label = td.Scalar(tf.int64)
    root_block = td.Record([('text', output_layer),
                            ('label', label)])

    # Turn dropout on for training, off for validation.
    #plan.train_feeds[keep_prob] = FLAGS.keep_prob

    return root_block


def example_generator(examples):
    for example in examples:
        yield {'text': example[1], 'label': example[0]}

    
if __name__ == "__main__":

    import argparse

    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input")
    parser.add_argument("--train", dest="train")
    parser.add_argument("--test", dest="test")
    parser.add_argument("--model", dest="model")
    parser.add_argument("--output", dest="output")
    
    parser.add_argument("--model_type", dest="model_type", default="flat", choices=["flat", "hierarchical"])
    parser.add_argument("--dev", dest="dev", type=float, default=.1)
    parser.add_argument("--attention", dest="attention", default="last", choices=["last", "mixture"])
    parser.add_argument("--char_embed_vector_length", dest="char_embed_vector_length", default=512, type=int)
    parser.add_argument("--char_state_vector_length", dest="char_state_vector_length", default=512, type=int)
    parser.add_argument("--word_state_vector_length", dest="word_state_vector_length", default=512, type=int)
    parser.add_argument("--char_cell_type", dest="char_cell_type", default="gru")
    parser.add_argument("--char_rnn_layer", dest="char_rnn_layer", default=2, type=int)
    parser.add_argument("--word_cell_type", dest="word_cell_type", default="gru", choices=["gru", "lstm", "gridlstm", "gridgru"])
    parser.add_argument("--word_rnn_layer", dest="word_rnn_layer", default=2, type=int)
    parser.add_argument("--dropout_probability", dest="dropout_probability", default=.5, type=float)
    parser.add_argument("--l2reg", dest="l2reg", default=False, action="store_true")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", default=10.0, type=float)
    parser.add_argument("--learning_rate", dest="learning_rate", default=.001, type=float)
    parser.add_argument("--optimizer", dest="optimizer", default="adam", choices=_OPTIMIZER_CLASSES.keys())
    options = parser.parse_args()

    
    td.define_plan_flags(default_plan_name='seal')
    FLAGS = tf.app.flags.FLAGS

    

    def setup_plan(plan, train, dev):
        if plan.mode != 'train': raise ValueError('only train mode supported')

        if FLAGS.hierarchical:
            print('Building hierarchical model...')
            root_block = create_hierarchical_model(nsym, ntag, plan)
        else:
            print('Building flat model...')
            root_block = create_flat_model(nsym, ntag, plan)

        plan.compiler = td.Compiler.create(root_block)

        logits, labels = plan.compiler.output_tensors
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels
        )

        global_step = tf.contrib.framework.get_or_create_global_step()
        lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                        100000, 0.96, staircase=True)

        optimizer = build_optimizer_from_params(FLAGS.optim, learning_rate=lr)
        tvars = tf.trainable_variables()

        nvars = np.prod(tvars[0].get_shape().as_list())
        for var in tvars[1:]:
            sh = var.get_shape().as_list()
            nvars += np.prod(sh)
        print("{} total variables".format(nvars))

        cost = None
        if FLAGS.l2reg:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars])
            cost = cross_entropy + l2_loss
            plan.losses['cross_entropy_L2'] = cost
        else:
            cost = cross_entropy
            plan.losses['cross_entropy'] = cost

        predictions = tf.argmax(logits, 1)
        plan.metrics['accuracy'] = tf.reduce_mean(
            tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
        )

        # A list of (gradient, variable) pairs. Variable is always
        # present, but gradient can be None.
        grads_and_vars = optimizer.compute_gradients(cost, tvars)
        
        tf.clip_by_global_norm([x[0] for x in grads_and_vars], FLAGS.max_grad_norm)

        plan.train_op = optimizer.apply_gradients(grads_and_vars,
                                                  global_step=global_step)

        plan.examples     = example_generator(train)
        plan.dev_examples = example_generator(valid)

    def main(_):
        pass
    #    random.seed(0)
    #    tf.set_random_seed(random.randint(0, 2**32))
        #assert 0 < FLAGS.keep_prob <= 1, '--keep_prob must be in (0, 1]\')'
    #    td.Plan.create_from_flags(setup_plan).run()

    #if __name__ == '__main__':
    #    tf.app.run()

    # training
    if options.train and options.output and options.input:


        
        sym_lookup, label_lookup = {"unk" : 0}, {"unk" : 0}
        train, dev = [], []
        with reader(gzip.open(options.train)) as ifd:
            indices = [int(l.strip()) for l in ifd]
        indices = set(indices)
        instances, labels = [], []        
        with reader(gzip.open(options.input)) as ifd:
            for i, line in enumerate(ifd):
                if i in indices:
                    cid, label, text = line.strip().split("\t")
                    label_lookup[label] = label_lookup.get(label, len(label_lookup))
                    syms = []
                    for c in text:
                        sym_lookup[c] = sym_lookup.get(c, len(sym_lookup))
                        syms.append(sym_lookup[c])
                    instances.append((label_lookup[label], syms))
        logging.info("Training with %d instances, %d labels", len(instances), len(label_lookup))
        
        #plan = td.Plan("train")
        plan = td.TrainPlan()
        plan.num_multiprocess_processes = 0
        plan.batch_size = 32
        plan.save_summaries_secs = 0
        plan.epochs = 2
        plan.batches_per_epoch = 8
        
        root_block = create_hierarchical_model(len(sym_lookup), len(label_lookup), options) if options.model_type == "hierarchical" else create_flat_model(len(sym_lookup), len(label_lookup), options)

        plan.compiler = td.Compiler.create(root_block)
        logits, labels = plan.compiler.output_tensors
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels
        )

        global_step = tf.contrib.framework.get_or_create_global_step()
        lr = tf.train.exponential_decay(options.learning_rate, global_step,
                                        100000, 0.96, staircase=True)

        optimizer = build_optimizer_from_params(options.optimizer, learning_rate=options.learning_rate)
        tvars = tf.trainable_variables()


        nvars = np.prod(tvars[0].get_shape().as_list())
        for var in tvars[1:]:
            sh = var.get_shape().as_list()
            nvars += np.prod(sh)
        print("{} total variables".format(nvars))

        cost = None
        if options.l2reg:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars])
            cost = cross_entropy + l2_loss
            plan.losses['cross_entropy_L2'] = cost
        else:
            cost = cross_entropy
            plan.losses['cross_entropy'] = cost

        predictions = tf.argmax(logits, 1)
        plan.metrics['accuracy'] = tf.reduce_mean(
            tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
        )

        # A list of (gradient, variable) pairs. Variable is always
        # present, but gradient can be None.
        grads_and_vars = optimizer.compute_gradients(cost, tvars)
        
        tf.clip_by_global_norm([x[0] for x in grads_and_vars], options.max_grad_norm)

        plan.train_op = optimizer.apply_gradients(grads_and_vars,
                                                  global_step=global_step)

        plan.examples     = example_generator(instances[0 : int(len(instances) * options.dev)])
        plan.dev_examples = example_generator(instances[int(len(instances) * options.dev):])
        plan.logdir = "temp/"
        #plan.batch_size = 32
        plan.finalize_stats()
        s = tf.train.Supervisor()
        #sess = 
        with s.managed_session() as sess:
            with sess.as_default():
                plan.run(s, sess)
                s = tf.train.Saver(tf.global_variables())
                print s.save(sess, options.output)


        #tf.app.run()
        #sup = plan.create_supervisor()
        
        #sess.run(plan)
        #plan.logdir = "temp/"
        #
        #plan.finalize_stats()
        #plan.run(supervisor=s)
        # if options.model_type == "hierarchical":
        #     print('Building hierarchical model...')
        #     root_block = create_hierarchical_model(len(sym_lookup), len(label_lookup)) #, plan)
        # else:
        #     print('Building flat model...')
        #     root_block = create_flat_model(len(sym_lookup), len(label_lookup)) #, plan)


        #with gzip.open(options.output, "w") as ofd:
        #    pickle.dump((classifier, dv, label_lookup), ofd)            

    # testing
    elif options.test and options.model and options.output and options.input:
        # with gzip.open(options.model) as ifd:
        #     classifier, dv, label_lookup = pickle.load(ifd)
        # with reader(gzip.open(options.test)) as ifd:
        #     indices = [int(l.strip()) for l in ifd]
        # indices = set(indices)
        # instances, gold = [], []
        # with reader(gzip.open(options.input)) as ifd:
        #     for i, line in enumerate(ifd):
        #         if i in indices:
        #             cid, label, text = line.strip().split("\t")
        #             instances.append(text)
        #             gold.append((cid, label))
        #logging.info("Testing with %d instances, %d labels", len(instances), len(label_lookup))
        #with writer(gzip.open(options.output, "w")) as ofd:
        #    ofd.write("\t".join(["ID", "GOLD"] + [inv_label_lookup[i] for i in range(len(inv_label_lookup))]) + "\n")
        #    for probs, (cid, gold) in zip(y, gold):
        #        ofd.write("\t".join([cid, gold] + ["%f" % x for x in probs.flatten()]) + "\n")
        pass
    else:
        print "ERROR: you must specify --input and --output, and either --train or --test and --model!"
