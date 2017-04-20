# SteamRoller ![Logo][logo.png]

This is a framework for testing various methods for performing different tasks in the broad area of "text classification".  It is designed to make it extremely easy to define new classification tasks, and new models, and drop them in to see how they perform.

## Getting started

### Requirements

The framework only requires two Python libraries:

* scons
* rpy2

And an R library:

* ggplot2

Of course, the models being tested have dependencies of their own.  To run the example, you will need:

* scikit-learn
* pip install https://storage.googleapis.com/tensorflow_fold/tensorflow_fold-0.0.1-cp27-none-linux_x86_64.whl

### Run the example

Copy the file `custom.py.template` to `custom.py`, and run:

```
scons -Q
```

to perform the example experiment (the model is Naive Bayes over character n-grams from n=1 to 3, the data is the English and Spanish tweets from the Twitter LID corpus).

## Adding your own experiments

### Define a task

The convention is that any file under `tasks/` represents a classification task, and should be a gzip-compressed text file with lines in the format `ID TAG TEXT`.  For example, you can look at `tasks/english_spanish_lid.txt.gz`, corresponding to a task called `English Spanish LID`.

### Instantiate a model

The convention for adding a new model is to place a script that performs training and classification under `models/`.  This script will be invoked twice for each experiment: the first time it will be passed four arguments: training, development, input, and output file names.  It should take these arguments and write a model to the output file.

The second time, it is also passed four arguments: model, test, input, and output file names.  It should apply the model to the test data, and write log-probabilities for each data point to the output file in format:

```ID TRUE_TAG LABEL1:LOGPROB1 LABEL2:LOGPROB2 ...```

Note that the train, dev, and test files are just gzipped lists of integers corresponding to lines in the input file: this is so we don't need to duplicate the data many times over.  See the model script `models/naive_bayes_wrapper.py` for an example of how they are used in combination with the input file.

### Modify custom.py

Aside from putting your data files and scripts in the right places, the only file you need to modify is `custom.py`.  Note that it isn't part of the actual git repository, it's meant to be where you define your own local experiments and customizations: `custom.py.template`, which *is* part of the repository, is a minimal example.  You will want to add entries to the `TASKS` or `MODELS` variables, as appropriate: see the comments in `custom.py.template` for more information.

## Advanced use

Since SCons is running each experiment by invoking user-defined scripts to train and apply models, it should have normal access to any resources (e.g. GPU) on the local machine.  Additionally, running SCons with the `-j N` option will allow up to `N` jobs to execute in parallel, which can significantly speed things up on a multi-core machine.  Work is under way on an SCons plugin that will allow experiments to be automatically mapped into a Grid Engine system for maximum throughput.

SCons is a very open-ended build system, and with Python being a dynamic language, it can be difficult to manage complexity.  So, you're encouraged to work in the patterns described above, adding experiments through the `custom.py` file.  If you run into a use-case that seems important but unsupported, feel free to contact me (tom@cs.jhu.edu).  If it represents a significant new class of experiments, maybe it warrants a new dedicated project!
